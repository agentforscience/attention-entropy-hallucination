"""Print markdown tables of the headline numbers from results/analysis/*.json."""
import json

import pandas as pd

TAGS = {"qwen25_7b": "Qwen2.5-7B", "llama31_8b": "Llama-3.1-8B"}
D = ["biography", "geography", "creative_works", "science", "trivia"]
R = {(t, l): json.load(open(f"results/analysis/{t}{'' if l == 'hybrid' else '_' + l}.json"))
     for t in TAGS for l in ["hybrid", "judge", "strmatch"]}
ci = lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]"
pc = lambda x: f"{x['rel_diff_pct']:+.1f}% [{x['rel_diff_ci95'][0]:+.1f}, {x['rel_diff_ci95'][1]:+.1f}]"

print("## Data\n\n| model | kept | abstain | domain: n / halluc rate |\n|---|---|---|---|")
for t, n in TAGS.items():
    r = R[t, "hybrid"]
    print(f"| {n} | {r['counts']['kept']} | {r['counts']['abstain']} | " + "; ".join(
        f"{d} {v['n']} / {v['halluc_rate']:.2f}" for d, v in r["by_domain_base"].items()) + " |")
    print(f"<!-- judge-strmatch agreement {r['judge_strmatch_agreement']:.3f} {r['judge_strmatch_agreement_by_domain']} ; prompt/answer len "
          + str({d: (round(v['n_prompt'], 1), round(v['n_answer'], 1)) for d, v in r['by_domain_base'].items()}) + " -->")

print("\n## E1 final-3-layer entropy, hallucinated vs factual (relative diff % [95% CI]; Cohen's d)\n")
for feat, lab in [("ent_ans", "mean over answer tokens"), ("ent_first", "first answer token"), ("nent_ans", "length-normalised, answer mean")]:
    print(f"\n**{lab}**\n\n| model | pooled | " + " | ".join(D) + " | macro |\n|---|---|" + "---|" * 6)
    for t, n in TAGS.items():
        e = R[t, "hybrid"]["E1_last3_effect"][feat]
        print(f"| {n} | " + " | ".join(f"{pc(e[k])}; d={e[k]['cohens_d']:+.2f}" for k in ["pooled"] + D) + f" | {e['macro_domain_rel_diff_pct']:+.1f}% |")
print("\n| model | pooled means (halluc / factual, nats) | OLS domain FE | OLS domain FE + length | layers sig. (FDR) |\n|---|---|---|---|---|")
for t, n in TAGS.items():
    r = R[t, "hybrid"]; e = r["E1_last3_effect"]
    f = lambda k: f"{e[k]['pct_of_factual_mean']:+.1f}% [{e[k]['ci95_pct'][0]:+.1f}, {e[k]['ci95_pct'][1]:+.1f}], p={e[k]['p']:.1e}"
    print(f"| {n} | {e['ent_ans']['pooled']['mean_halluc']:.3f} / {e['ent_ans']['pooled']['mean_factual']:.3f} | {f('ols_domain_FE')} | {f('ols_domain_FE_plus_length')} | {r['per_layer_n_sig_fdr']['ent_ans']}/{r['n_layers']} |")
print("\nLabel robustness (ent_ans, pooled / macro):")
for t, n in TAGS.items():
    print(f"- {n}: " + "; ".join(f"{l}: {pc(R[t, l]['E1_last3_effect']['ent_ans']['pooled'])} / {R[t, l]['E1_last3_effect']['ent_ans']['macro_domain_rel_diff_pct']:+.1f}%" for l in ["hybrid", "judge", "strmatch"]))

print("\n## E2/E4 pooled classifiers (AUROC [95% CI] / within-domain macro AUROC of the same scores / AUPRC)\n")
names = list(R["llama31_8b", "hybrid"]["E2_pooled_classifiers"])
print("| features | " + " | ".join(TAGS.values()) + " |\n|---|---|---|")
for nm in names:
    print(f"| {nm} | " + " | ".join(
        (lambda v: f"{v['auroc']:.3f} {ci(v['ci95'])} / {v['within_domain_macro_auroc']:.3f} / {v['auprc']:.3f}")(R[t, "hybrid"]["E2_pooled_classifiers"][nm]) for t in TAGS) + " |")
print("\nSeed sensitivity (SD of AUROC over 5 CV repeats), entropy_layers:",
      {t: round(R[t, 'hybrid']['E2_pooled_classifiers']['entropy_layers (ans mean)']['repeat_sd'], 4) for t in TAGS})
print("\nPaired differences:")
for t, n in TAGS.items():
    for k, v in R[t, "hybrid"]["E4_paired"].items():
        print(f"- {n}: {k}: {v['diff']:+.3f} {ci(v['ci95'])}, p={v['p_two_sided']}")
print("\nLabel robustness, entropy_layers (ans mean) pooled / within-domain macro:")
for t, n in TAGS.items():
    print(f"- {n}: " + "; ".join((lambda v: f"{l}: {v['auroc']:.3f} / {v['within_domain_macro_auroc']:.3f}")(R[t, l]["E2_pooled_classifiers"]["entropy_layers (ans mean)"]) for l in ["hybrid", "judge", "strmatch"]))

print("\n## E3 within-domain AUROC (trained and cross-validated inside each domain)\n")
for t, n in TAGS.items():
    e = R[t, "hybrid"]["E3_within_domain"]
    print(f"\n**{n}**\n\n| features | " + " | ".join(D) + " | mean |\n|---|" + "---|" * 6)
    for nm in e["science"]:
        print(f"| {nm} | " + " | ".join(f"{e[d][nm]['auroc']:.3f} {ci(e[d][nm]['ci95'])}" for d in D) + f" | {sum(e[d][nm]['auroc'] for d in D) / 5:.3f} |")
    o = R[t, "hybrid"]["E3_ordering"]
    print(f"\nranking: {' > '.join(o['ranking_high_to_low'])}; bio - science = {o['bio_minus_science']['diff']:+.3f} {ci(o['bio_minus_science']['ci95'])}, p={o['bio_minus_science']['p_two_sided']}; "
          f"P(bio highest)={o['p_boot_bio_highest']:.3f}, P(science lowest)={o['p_boot_science_lowest']:.3f}; transfer off-diagonal mean {R[t, 'hybrid']['E3_transfer_offdiag_mean']:.3f}")
    for l in ["judge", "strmatch"]:
        o = R[t, l]["E3_ordering"]; e2 = R[t, l]["E3_within_domain"]
        print(f"  [{l}] ranking: {' > '.join(o['ranking_high_to_low'])}; " + ", ".join(f"{d} {e2[d]['entropy_layers (ans mean)']['auroc']:.3f}" for d in D))
    print("\ntransfer (rows train, cols test):\n")
    print(pd.read_csv(f"results/analysis/{t}_transfer.csv", index_col=0).round(3).to_markdown())
    print("\nby PopQA relation:\n")
    print(pd.read_csv(f"results/analysis/{t}_by_relation.csv").round(3).to_markdown(index=False))
