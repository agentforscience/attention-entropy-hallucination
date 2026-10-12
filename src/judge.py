"""Label each generated answer as correct / incorrect (hallucinated) / abstain.

Two labels are written to results/evaluations/<tag>_labels.jsonl:
  judge  : reference-based LLM judge (Qwen3-8B, thinking off, greedy) - primary label
  strmatch: 1 if any normalized gold alias is contained in the normalized answer
"""
import argparse
import json
import os
import re
import string
import time
import unicodedata

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_SYSTEM = "You are a strict but fair grader of short factual answers."
JUDGE_TEMPLATE = """Question: {question}
Reference answer(s): {gold}
Model answer: {answer}

Grade the model answer against the reference answer(s).
- CORRECT: it states the same fact as at least one reference answer (different wording, spelling variants, extra non-contradictory detail, or a more specific answer that implies the reference are all fine).
- INCORRECT: it states a different fact, or is too vague/general to match the reference.
- ABSTAIN: it declines to answer or says it does not know.
Reply with exactly one word: CORRECT, INCORRECT, or ABSTAIN."""


def normalize(s):
    """SQuAD-style normalization plus accent stripping."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = "".join(c if c not in string.punctuation else " " for c in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def string_match(answer, gold):
    a = normalize(answer)
    return int(any(g and re.search(rf"\b{re.escape(g)}\b", a) for g in map(normalize, gold)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--judge", default="Qwen/Qwen3-8B")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--headroom", type=float, default=6, help="GiB left free on each GPU")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.judge)
    tok.padding_side = "left"
    # The GPUs are shared with other jobs: spread the judge over whatever is free right now,
    # leaving headroom, and checkpoint every batch so an OOM never loses work.
    free = {i: torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(torch.cuda.device_count())}
    max_memory = {i: f"{int(f - args.headroom)}GiB" for i, f in free.items() if f > args.headroom + 2}
    print("free GiB", free, "max_memory", max_memory, flush=True)
    kw = {}
    if args.load_4bit:  # fallback when the shared GPUs cannot hold the bf16 judge (28 GB)
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.judge, dtype=torch.bfloat16, device_map="auto",
                                                 max_memory=max_memory, **kw).eval()
    first_dev = next(model.parameters()).device
    os.makedirs("results/evaluations", exist_ok=True)

    for tag in args.tags:
        recs = [json.loads(l) for l in open(f"results/model_outputs/{tag}.jsonl")]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": JUDGE_SYSTEM},
             {"role": "user", "content": JUDGE_TEMPLATE.format(
                 question=r["question"], gold=" | ".join(r["gold"][:12]), answer=r["answer"] or "(empty)")}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in recs]  # flag only matters for Qwen3
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        verdicts = [None] * len(recs)
        ckpt = f"results/evaluations/{tag}_judge_ckpt.jsonl"
        if os.path.exists(ckpt):
            for l in open(ckpt):
                i, v = json.loads(l)
                verdicts[i] = v
        order = [i for i in order if verdicts[i] is None]
        ck = open(ckpt, "a")
        for b in tqdm(range(0, len(order), args.batch_size), mininterval=30):
            idx = order[b: b + args.batch_size]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(first_dev)
            for attempt in range(20):
                try:
                    with torch.no_grad():
                        out = model.generate(**enc, do_sample=False, max_new_tokens=4, pad_token_id=tok.pad_token_id)
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    time.sleep(15)
            else:
                raise RuntimeError("judge: repeated OOM")
            for i, o in zip(idx, out):
                v = tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
                verdicts[i] = ("INCORRECT" if v.startswith("INCORRECT") else "CORRECT" if v.startswith("CORRECT")
                               else "ABSTAIN" if v.startswith("ABSTAIN") else "UNPARSED:" + v)
                ck.write(json.dumps([i, verdicts[i]]) + "\n")
            ck.flush()
        ck.close()
        with open(f"results/evaluations/{tag}_labels.jsonl", "w") as f:
            for r, v in zip(recs, verdicts):
                f.write(json.dumps(dict(qid=r["qid"], judge=v, strmatch=string_match(r["answer"], r["gold"]))) + "\n")
        from collections import Counter
        print(tag, Counter(verdicts), "strmatch acc", sum(string_match(r["answer"], r["gold"]) for r in recs) / len(recs))


if __name__ == "__main__":
    main()
