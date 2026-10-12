"""Generate closed-book answers and extract attention-entropy + logit features.

For each question:
  1. greedy-decode a short answer (chat template, fixed system prompt);
  2. run ONE teacher-forced pass over prompt+answer with eager attention;
  3. for every layer/head, compute the Shannon entropy (nats) of the attention row of
     each position that *produces* an answer token.

Saved to results/features/<tag>.npz (arrays indexed like results/model_outputs/<tag>.jsonl):
  ent_ans   (N, L, H) mean entropy over answer-producing positions
  ent_first (N, L, H) entropy at the position producing the first answer token
  nent_ans, nent_first: same, each row divided by log(#attended positions)
  logit stats and lengths in the jsonl.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = ("You are a knowledgeable assistant. Answer the question with only the short answer "
          "(a name or a few words). Do not explain. Always give your best guess.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="datasets/processed/questions.jsonl")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    qs = [json.loads(l) for l in open(args.data)]
    if args.limit:
        qs = qs[:: max(1, len(qs) // args.limit)][: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="eager").cuda().eval()
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    print(f"{args.model}: L={L} H={H}", flush=True)
    stop_ids = {tok.eos_token_id, *[i for i in tok.convert_tokens_to_ids(["<|eot_id|>", "<|im_end|>"])
                                    if i is not None and i != tok.unk_token_id]}

    os.makedirs("results/features", exist_ok=True)
    os.makedirs("results/model_outputs", exist_ok=True)
    feats = {k: [] for k in ["ent_ans", "ent_first", "nent_ans", "nent_first"]}
    records = []
    t0 = time.time()
    # Sort by length so batches need little padding; original order restored at the end.
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q["question"]}],
        tokenize=False, add_generation_prompt=True) for q in qs]
    p_ids = [tok(p, add_special_tokens=False).input_ids for p in prompts]
    order = np.argsort([len(x) for x in p_ids], kind="stable")

    # Variable batch size: attention maps cost O(B * T^2) per layer, so long prompts get small batches.
    batches, i = [], 0
    while i < len(order):
        T_max = len(p_ids[order[i]]) + args.max_new_tokens
        bs = max(1, min(args.batch_size, int(args.batch_size * (100 / T_max) ** 2)))
        batches.append(order[i: i + bs])
        i += bs
    for b, idx in enumerate(tqdm(batches, mininterval=30)):
        batch = [p_ids[i] for i in idx]
        T = max(map(len, batch))
        ids = torch.full((len(batch), T), tok.pad_token_id)
        mask = torch.zeros((len(batch), T), dtype=torch.long)
        for j, x in enumerate(batch):
            ids[j, T - len(x):] = torch.tensor(x)
            mask[j, T - len(x):] = 1
        with torch.no_grad():
            out = model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), do_sample=False,
                                 max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id,
                                 eos_token_id=list(stop_ids))
        gens = []
        for j in range(len(batch)):
            g = out[j, T:].tolist()
            cut = next((k for k, t in enumerate(g) if t in stop_ids), len(g))
            gens.append(g[:cut])

        # Teacher-forced pass over prompt + answer (answer tokens only, no EOS).
        full = [batch[j] + gens[j] for j in range(len(batch))]
        T2 = max(map(len, full))
        ids2 = torch.full((len(full), T2), tok.pad_token_id)
        mask2 = torch.zeros((len(full), T2), dtype=torch.long)
        for j, x in enumerate(full):
            ids2[j, T2 - len(x):] = torch.tensor(x)
            mask2[j, T2 - len(x):] = 1
        with torch.no_grad():
            o = model(input_ids=ids2.cuda(), attention_mask=mask2.cuda(), output_attentions=True)
            # (L, B, H, T2): entropy of every attention row.
            ent = torch.stack([-(a.float() * torch.log(a.float().clamp_min(1e-12))).sum(-1)
                               for a in o.attentions])
            if b == 0:  # sanity: rows are probability distributions
                s = o.attentions[-1].float().sum(-1)[mask2.bool().cuda()[:, None, :].expand(-1, H, -1)]
                assert (s - 1).abs().max() < 2e-2, s
            logp = torch.log_softmax(o.logits.float(), -1)
            pent = -(logp.exp() * logp).sum(-1)  # predictive entropy per position (B, T2)

        for j, qi in enumerate(idx):
            P, m, n = len(batch[j]), len(gens[j]), len(full[j])
            off = T2 - n
            rec = dict(qs[qi], answer=tok.decode(gens[j], skip_special_tokens=True).strip(),
                       n_prompt=P, n_answer=m, order=int(qi))
            if m == 0:
                for k in feats:
                    feats[k].append(np.full((L, H), np.nan, dtype=np.float32))
                records.append(rec)
                continue
            pos = torch.arange(P - 1, P + m - 1, device=ent.device)  # unpadded positions producing answer tokens
            e = ent[:, j, :, off + pos]                               # (L, H, m)
            ne = e / torch.log((pos + 1).float())                     # row at position p attends to p+1 tokens
            feats["ent_ans"].append(e.mean(-1).cpu().numpy())
            feats["ent_first"].append(e[..., 0].cpu().numpy())
            feats["nent_ans"].append(ne.mean(-1).cpu().numpy())
            feats["nent_first"].append(ne[..., 0].cpu().numpy())
            tgt = torch.tensor(gens[j], device=ent.device)
            nll = -logp[j, off + pos, tgt]
            pe = pent[j, off + pos]
            rec.update(nll_mean=nll.mean().item(), nll_max=nll.max().item(), nll_first=nll[0].item(),
                       pent_mean=pe.mean().item(), pent_max=pe.max().item(), pent_first=pe[0].item())
            records.append(rec)
        del o, ent, logp, pent

    inv = np.argsort([r["order"] for r in records])
    records = [records[i] for i in inv]
    with open(f"results/model_outputs/{args.tag}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.savez_compressed(f"results/features/{args.tag}.npz",
                        **{k: np.stack(v)[inv].astype(np.float16) for k, v in feats.items()})
    print(f"done {len(records)} examples in {time.time() - t0:.0f}s; "
          f"empty answers: {sum(r['n_answer'] == 0 for r in records)}", flush=True)


if __name__ == "__main__":
    main()
