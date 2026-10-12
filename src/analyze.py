"""Statistical analysis for the attention-entropy hallucination study.

Usage: python src/analyze.py --tag qwen25_7b [--label hybrid|judge|strmatch]
Labels: hybrid (primary) = correct if a gold alias string-matches OR the LLM judge says CORRECT.
Writes results/analysis/<tag>[_<label>].json and CSV tables next to it.
"""
import argparse
import json
import os
import re
import time
import warnings

# The machine is shared and heavily loaded: avoid BLAS thread oversubscription.
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

SEED = 42
# Rule-based abstention detector applied on top of the judge: the 8B judge sometimes grades
# "Unknown" / "N/A" as INCORRECT (or even CORRECT) instead of ABSTAIN.
ABSTAIN_RE = re.compile(
    r"(?i)^\W*(unknown|none|n/a|not (known|available|applicable|specified|sure|found)|"
    r"no (answer|information|data|record)s?( available| found)?|unclear|unspecified|undetermined)\W*$|"
    r"couldn'?t find|could not find|no (public(ly available)? |reliable |specific |available )?information|"
    r"(don'?t|do not) (know|have)|not (enough|sufficient) information|unable to|"
    r"cannot (determine|find|provide|verify)|can'?t (determine|find|provide|verify)|i'm not (sure|aware)|"
    r"is not a (person|human)|^\W*tbd\W*$|no single (author|director)|not aware of|no historical|not publicly (known|available)|is not (known|specified|available|documented)")
DOMAINS = ["biography", "geography", "creative_works", "science", "trivia"]
LOGIT_COLS = ["nll_mean", "nll_max", "nll_first", "pent_mean", "pent_max", "pent_first"]


def load(tag, label="hybrid", eval_dir="results/evaluations"):
    """Return (df, feats) restricted to answered, gradable examples. y=1 means hallucinated."""
    df = pd.DataFrame([json.loads(l) for l in open(f"results/model_outputs/{tag}.jsonl")])
    lab = pd.DataFrame([json.loads(l) for l in open(f"{eval_dir}/{tag}_labels.jsonl")])
    assert (df.qid.values == lab.qid.values).all()
    df["judge"], df["strmatch"] = lab.judge.values, lab.strmatch.values
    npz = np.load(f"results/features/{tag}.npz")
    feats = {k: npz[k].astype(np.float64) for k in npz.files}
    # Abstentions are defined by the judge under both labelings, so both use the same examples.
    rule_abstain = df.answer.apply(lambda a: bool(ABSTAIN_RE.search(a)))
    # Abstention: rule match, or judge says ABSTAIN and no gold alias appears in the answer.
    abstain = rule_abstain | ((df.judge == "ABSTAIN") & (df.strmatch == 0))
    unparsed = df.judge.str.startswith("UNPARSED") & (df.strmatch == 0)
    keep = (df.n_answer > 0) & ~abstain & ~unparsed  # identical example set for all three labelings
    counts = dict(total=len(df), empty=int((df.n_answer == 0).sum()), rule_abstain=int(rule_abstain.sum()),
                  abstain=int(abstain.sum()), unparsed=int(unparsed.sum()), kept=int(keep.sum()))
    judge_ok = (df.judge == "CORRECT").astype(int)
    correct = {"hybrid": ((df.strmatch == 1) | (judge_ok == 1)).astype(int), "judge": judge_ok, "strmatch": df.strmatch}[label]
    df["judge_correct"] = judge_ok
    df["y"] = 1 - correct
    idx = np.where(keep.values)[0]
    df = df.iloc[idx].reset_index(drop=True)
    feats = {k: v[idx] for k, v in feats.items()}
    assert not any(np.isnan(v).any() for v in feats.values())
    return df, feats, counts


def boot_ci(fn, n, B, rng, strata=None):
    """Percentile bootstrap CI of fn(indices); resamples within strata if given."""
    groups = [np.arange(n)] if strata is None else [np.where(strata == s)[0] for s in np.unique(strata)]
    vals = []
    for _ in range(B):
        ii = np.concatenate([rng.choice(g, len(g)) for g in groups])
        vals.append(fn(ii))
    return np.percentile(vals, [2.5, 97.5]).tolist(), np.asarray(vals)


def cohens_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp)


def fast_auc(y, s):
    """Rank-based AUROC (ties get average rank)."""
    r = stats.rankdata(s)
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def boot_means(a, B, rng, chunk=500):
    """Bootstrap distribution of the mean of a (vectorised)."""
    return np.concatenate([a[rng.integers(0, len(a), (min(chunk, B - i), len(a)))].mean(1) for i in range(0, B, chunk)])


def effect(x, y, rng, B=10000):
    """Hallucinated-vs-factual comparison of a scalar x. Bootstrap resamples each class independently."""
    h, f = x[y == 1], x[y == 0]
    rel = (boot_means(h, B, rng) / boot_means(f, B, rng) - 1) * 100
    return dict(n_halluc=int(len(h)), n_factual=int(len(f)), mean_halluc=float(h.mean()),
                mean_factual=float(f.mean()), rel_diff_pct=float((h.mean() / f.mean() - 1) * 100),
                rel_diff_ci95=np.percentile(rel, [2.5, 97.5]).tolist(), cohens_d=cohens_d(h, f),
                mwu_p=float(stats.mannwhitneyu(h, f, alternative="two-sided").pvalue),
                auroc_single=fast_auc(y, x))


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def cv_scores(X, y, n_rep=5):
    """Out-of-fold decision scores averaged over repeats, plus the per-repeat AUROCs."""
    X = X.reshape(len(X), -1)
    oof = np.zeros((n_rep, len(y)))
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=SEED)
    for k, (tr, te) in enumerate(rskf.split(X, y)):
        oof[k // 5, te] = clf().fit(X[tr], y[tr]).decision_function(X[te])
    rep_auc = [roc_auc_score(y, o) for o in oof]
    return oof.mean(0), rep_auc


def auc_report(score, y, rng, rep_auc=None, B=2000):
    ci, vals = boot_ci(lambda ii: fast_auc(y[ii], score[ii]), len(y), B, rng, strata=y)
    out = dict(auroc=float(roc_auc_score(y, score)), ci95=ci, auprc=float(average_precision_score(y, score)),
               n=int(len(y)), halluc_rate=float(y.mean()))
    if rep_auc is not None:
        out["repeat_mean"], out["repeat_sd"] = float(np.mean(rep_auc)), float(np.std(rep_auc, ddof=1))
    return out, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--label", default="hybrid", choices=["hybrid", "judge", "strmatch"])
    ap.add_argument("--eval_dir", default="results/evaluations")
    ap.add_argument("--out_dir", default="results/analysis")
    ap.add_argument("--fast", action="store_true", help="skip per-head model and transfer matrix")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    T0 = time.time()
    rng = np.random.default_rng(SEED)
    df, F, counts = load(args.tag, args.label, args.eval_dir)
    y, dom = df.y.values, df.domain.values
    L = F["ent_ans"].shape[1]
    res = dict(tag=args.tag, label=args.label, counts=counts, n_layers=L, n_heads=F["ent_ans"].shape[2])
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    sfx = "" if args.label == "hybrid" else "_" + args.label

    res["by_domain_base"] = {d: dict(n=int((dom == d).sum()), halluc_rate=float(y[dom == d].mean()),
                                     n_prompt=float(df.n_prompt[dom == d].mean()),
                                     n_answer=float(df.n_answer[dom == d].mean())) for d in DOMAINS}
    res["judge_strmatch_agreement"] = float((df.judge_correct == df.strmatch).mean())
    res["judge_strmatch_agreement_by_domain"] = {d: float((df.judge_correct == df.strmatch)[dom == d].mean()) for d in DOMAINS}

    # Layer-level features: mean over heads.
    lay = {k: v.mean(2) for k, v in F.items()}  # (N, L)
    last3 = {k: v[:, -3:].mean(1) for k, v in lay.items()}
    logn_p, logn_a = np.log(df.n_prompt.values), np.log(df.n_answer.values)

    # ---------------- E1: effect size ----------------
    e1 = {}
    for k in ["ent_ans", "ent_first", "nent_ans", "nent_first"]:
        e1[k] = dict(pooled=effect(last3[k], y, rng))
        for d in DOMAINS:
            m = dom == d
            e1[k][d] = effect(last3[k][m], y[m], rng, B=4000)
        e1[k]["macro_domain_rel_diff_pct"] = float(np.mean([e1[k][d]["rel_diff_pct"] for d in DOMAINS]))
    # Adjusted difference: OLS with domain fixed effects and length covariates (HC3 SEs).
    d_ols = pd.DataFrame(dict(x=last3["ent_ans"], y=y, lp=logn_p, la=logn_a, dom=dom))
    for name, formula in [("domain_FE", "x ~ y + C(dom)"), ("domain_FE_plus_length", "x ~ y + C(dom) + lp + la")]:
        fit = smf.ols(formula, d_ols).fit(cov_type="HC3")
        base = d_ols.x[d_ols.y == 0].mean()
        lo, hi = fit.conf_int().loc["y"]
        e1[f"ols_{name}"] = dict(coef_nats=float(fit.params["y"]), pct_of_factual_mean=float(fit.params["y"] / base * 100),
                                 ci95_pct=[float(lo / base * 100), float(hi / base * 100)], p=float(fit.pvalues["y"]))
    res["E1_last3_effect"] = e1

    # Per-layer profile with BH-FDR over layers.
    rows = []
    for k in ["ent_ans", "ent_first"]:
        for l in range(L):
            x = lay[k][:, l]
            h, f = x[y == 1], x[y == 0]
            rows.append(dict(feature=k, layer=l, mean_halluc=h.mean(), mean_factual=f.mean(),
                             rel_diff_pct=(h.mean() / f.mean() - 1) * 100, cohens_d=cohens_d(h, f),
                             d_within_domain=float(np.mean([cohens_d(x[(y == 1) & (dom == d)], x[(y == 0) & (dom == d)])
                                                            for d in DOMAINS])),
                             p=stats.mannwhitneyu(h, f).pvalue))
    pl = pd.DataFrame(rows)
    pl["p_fdr"] = np.concatenate([multipletests(g.p, method="fdr_bh")[1] for _, g in pl.groupby("feature", sort=False)])
    pl.to_csv(f"{out_dir}/{args.tag}{sfx}_per_layer.csv", index=False)
    res["per_layer_n_sig_fdr"] = {k: int((g.p_fdr < 0.05).sum()) for k, g in pl.groupby("feature")}

    print(f"E1 done {time.time() - T0:.0f}s", flush=True)
    # ---------------- E2/E4: classifiers (pooled) ----------------
    length = np.c_[logn_p, logn_a]
    logit = df[LOGIT_COLS].values
    dom_oh = pd.get_dummies(df.domain).values.astype(float)
    sets = {
        "domain_only": dom_oh,
        "length_only": length,
        "entropy_layers + domain": np.c_[lay["ent_ans"], dom_oh],
        "entropy_last3_mean (1 feat)": last3["ent_ans"][:, None],
        "entropy_layers (ans mean)": lay["ent_ans"],
        "entropy_layers (first token)": lay["ent_first"],
        "norm_entropy_layers (ans mean)": lay["nent_ans"],
        "entropy_layers + length": np.c_[lay["ent_ans"], length],
        "logit_baseline": logit,
        "nll_mean only": df[["nll_mean"]].values,
        "entropy_layers + logit": np.c_[lay["ent_ans"], logit],
    }
    if not args.fast:
        sets["entropy_heads (L*H)"] = F["ent_ans"]
    e2, oof = {}, {}
    for name, X in sets.items():
        s, rep = cv_scores(X, y)
        oof[name] = s
        e2[name], _ = auc_report(s, y, rng, rep)
        # AUROC of the pooled model's scores inside each domain (removes between-domain base-rate signal).
        e2[name]["within_domain_macro_auroc"] = float(np.mean([roc_auc_score(y[dom == d], s[dom == d]) for d in DOMAINS]))
        print(f"{name:34s} AUROC {e2[name]['auroc']:.3f} {np.round(e2[name]['ci95'], 3)} "
              f"within-domain macro {e2[name]['within_domain_macro_auroc']:.3f}", flush=True)
    res["E2_pooled_classifiers"] = e2
    # Paired bootstrap: does adding entropy to logits help? does entropy beat length?
    def paired(a, b, B=2000):
        ci, v = boot_ci(lambda ii: fast_auc(y[ii], oof[a][ii]) - fast_auc(y[ii], oof[b][ii]), len(y), B, rng, strata=y)
        return dict(diff=float(roc_auc_score(y, oof[a]) - roc_auc_score(y, oof[b])), ci95=ci, p_two_sided=float(min(1, 2 * min((v <= 0).mean(), (v >= 0).mean()))))
    res["E4_paired"] = {
        "entropy_layers - length_only": paired("entropy_layers (ans mean)", "length_only"),
        "entropy_layers - domain_only": paired("entropy_layers (ans mean)", "domain_only"),
        "entropy_layers - logit_baseline": paired("entropy_layers (ans mean)", "logit_baseline"),
        "entropy+logit - logit_baseline": paired("entropy_layers + logit", "logit_baseline"),
    }

    print(f"E2/E4 done {time.time() - T0:.0f}s", flush=True)
    # ---------------- E3: per-domain ----------------
    e3, boots = {}, {}
    for d in DOMAINS:
        m = dom == d
        e3[d] = {}
        for name in ["entropy_layers (ans mean)", "entropy_layers (first token)", "norm_entropy_layers (ans mean)",
                     "length_only", "logit_baseline", "entropy_layers + logit"]:
            s, rep = cv_scores(sets[name][m], y[m])
            e3[d][name], v = auc_report(s, y[m], rng, rep)
            if name == "entropy_layers (ans mean)":
                boots[d] = v
    res["E3_within_domain"] = e3
    key = "entropy_layers (ans mean)"
    order = sorted(DOMAINS, key=lambda d: -e3[d][key]["auroc"])
    diff = boots["biography"] - boots["science"]
    res["E3_ordering"] = dict(
        ranking_high_to_low=order,
        bio_minus_science=dict(diff=e3["biography"][key]["auroc"] - e3["science"][key]["auroc"],
                               ci95=np.percentile(diff, [2.5, 97.5]).tolist(),
                               p_two_sided=float(min(1, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))),
        # Bootstrap probability that biography is the top domain and science the bottom one.
        p_boot_bio_highest=float(np.mean(np.argmax(np.c_[[boots[d] for d in DOMAINS]], 0) == DOMAINS.index("biography"))),
        p_boot_science_lowest=float(np.mean(np.argmin(np.c_[[boots[d] for d in DOMAINS]], 0) == DOMAINS.index("science"))))
    if not args.fast:
        T = np.zeros((5, 5))
        for i, a in enumerate(DOMAINS):
            for j, b in enumerate(DOMAINS):
                if i == j:
                    T[i, j] = e3[a][key]["auroc"]
                else:
                    T[i, j] = roc_auc_score(y[dom == b], clf().fit(lay["ent_ans"][dom == a], y[dom == a])
                                            .decision_function(lay["ent_ans"][dom == b]))
        pd.DataFrame(T, index=DOMAINS, columns=DOMAINS).to_csv(f"{out_dir}/{args.tag}{sfx}_transfer.csv")
        res["E3_transfer_offdiag_mean"] = float(T[~np.eye(5, dtype=bool)].mean())
        # PopQA-relation breakdown (error analysis): pooled-model scores within each relation.
        rel = []
        for r, g in df[df.source == "PopQA"].groupby("relation"):
            if g.y.nunique() == 2 and g.y.value_counts().min() >= 20:
                rel.append(dict(relation=r, domain=g.domain.iloc[0], n=len(g), halluc_rate=g.y.mean(),
                                auroc_entropy=roc_auc_score(g.y, oof[key][g.index]),
                                auroc_logit=roc_auc_score(g.y, oof["logit_baseline"][g.index])))
        pd.DataFrame(rel).to_csv(f"{out_dir}/{args.tag}{sfx}_by_relation.csv", index=False)

    # Save per-example scores for figures / error analysis.
    out = df[["qid", "domain", "relation", "question", "answer", "y", "n_prompt", "n_answer", "nll_mean"]].copy()
    out["ent_last3"] = last3["ent_ans"]
    out["score_entropy"] = oof[key]
    out["score_logit"] = oof["logit_baseline"]
    out.to_csv(f"{out_dir}/{args.tag}{sfx}_examples.csv", index=False)
    np.save(f"{out_dir}/{args.tag}{sfx}_layer_ent.npy", lay["ent_ans"])
    json.dump(res, open(f"{out_dir}/{args.tag}{sfx}.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ["counts", "by_domain_base", "judge_strmatch_agreement", "E3_ordering", "E4_paired"]}, indent=1))
    print(json.dumps({k: {kk: (round(vv["rel_diff_pct"], 2), np.round(vv["rel_diff_ci95"], 2).tolist(), round(vv["cohens_d"], 3))
                          for kk, vv in v.items() if isinstance(vv, dict)} for k, v in e1.items() if not k.startswith("ols")}, indent=1))
    print({k: v for k, v in e1.items() if k.startswith("ols")})
    print({d: (round(e3[d][key]["auroc"], 3), np.round(e3[d][key]["ci95"], 3).tolist()) for d in DOMAINS})


if __name__ == "__main__":
    main()
