# Planning: Attention Entropy as a Predictor of Hallucination

Written before any model was run. Deviations are logged at the bottom.

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs state wrong facts in the same fluent register as right ones. Detectors that need
only a single forward pass and no external knowledge are attractive because they are
cheap enough to run on every response. Attention maps are available for free in that
pass; if their dispersion (entropy) tracks factual error, it is a nearly zero-cost
warning signal and also says something about *how* a model behaves when it lacks the fact.

### Gap in Existing Work
(The resource-finding stage was skipped for this run, so there is no
`literature_review.md`; this is from my own knowledge of the area.)
- Output-distribution methods (token likelihood, predictive/semantic entropy,
  SelfCheckGPT) use logits or multiple samples, not attention.
- Hidden-state probes (SAPLMA; INSIDE/EigenScore; Orgad et al. 2024) use residual activations.
- Attention-based detectors exist but use other statistics: Lookback Lens (ratio of
  attention on context vs. generated tokens, RAG setting), LLM-Check (attention kernel
  log-determinants), Yuksekgonul et al. 2024 (attention *mass* on constraint tokens).
- What is missing is a direct, controlled test of the simple claim "attention is more
  diffuse when hallucinating": a stated effect size by layer, a classifier restricted to
  layer-wise entropy, explicit control of the sequence-length confound (entropy is
  bounded by log n), and a per-domain breakdown.

### Our Novel Contribution
A pre-specified test of three quantitative claims (15-30% higher late-layer entropy;
AUROC > 0.75 from layer-wise entropy alone; biography > science ordering) on two
open-weight model families across five knowledge domains, with length controls and
logit baselines.

### Experiment Justification
- E1 (effect size by layer): directly tests the "15-30% higher in final 3 layers" claim.
- E2 (entropy-only logistic classifier): tests "AUROC > 0.75".
- E3 (per-domain AUROC + cross-domain transfer): tests the domain-ordering claim and
  whether the signal is domain-general.
- E4 (controls/baselines): length-only, token-likelihood, and combined models; tells
  us whether entropy carries signal beyond trivial confounds and beyond logits.
- E5 (robustness): second model family, alternative labels, first-token vs
  answer-averaged features, per-head features.

## Research Question
In closed-book short-answer QA, is self-attention entropy at answer-generation time
higher for hallucinated (factually wrong) answers than for correct ones, by how much in
the last 3 layers, and how well does a logistic classifier on per-layer entropy detect
hallucination, overall and per domain?

## Hypothesis Decomposition
- H1: mean attention entropy over the final 3 layers is 15-30% higher for hallucinated
  than for correct answers. Supported only if the relative difference (and its 95% CI)
  lies in/overlaps that band with the right sign.
- H2: logistic regression on per-layer entropy features only reaches cross-validated
  AUROC > 0.75 (pooled over domains). Supported if the 95% CI lower bound > 0.75;
  "borderline" if the CI straddles 0.75.
- H3: within-domain AUROC is highest for biography and lowest for science.

Variables. Independent: answer correctness (hallucinated vs factual), domain, layer,
model. Dependent: attention entropy; detector AUROC.

## Proposed Methodology

### Models (local, open weights; attention maps are required, so no API models)
- Qwen2.5-7B-Instruct (28 layers x 28 heads) - primary.
- Llama-3.1-8B-Instruct (32 layers x 32 heads) - replication.
fp16, greedy decoding, max 24 new tokens, chat template with a fixed system prompt.

### Data (1,500 questions per domain, seed 42, 7,500 per model)
| Domain | Source |
|---|---|
| biography | PopQA relations occupation / place of birth / father / mother |
| geography | PopQA country / capital / capital of |
| creative works | PopQA author / director / producer / screenwriter / composer |
| science | SciQ (train split, question + correct answer, no support passage) |
| general trivia | TriviaQA rc.nocontext validation |

Known confound: science and trivia come from different datasets than the three PopQA
domains, so "domain" is partly "dataset/question style". Reported as a limitation.

### Labels
- Primary: reference-based LLM judge (Qwen2.5-14B-Instruct, a different and larger
  model than either subject, temperature 0) given question, gold answers, model answer;
  outputs CORRECT / INCORRECT / ABSTAIN.
- Secondary (robustness): normalized string containment of any gold alias.
- Abstentions/refusals are excluded from the hallucinated-vs-factual comparison and counted.
- Rationale for a judge as primary: SciQ gold answers are free-form phrases where
  string match would produce false "hallucinations" specifically in the science domain,
  biasing H3. No commercial API key is available in this environment, hence a local judge.

### Features
One teacher-forced forward pass over prompt + generated answer with eager attention.
For every layer l, head h, and each position p that *produces* an answer token:
H = -sum_j a_pj log a_pj (nats). Aggregations saved per example:
- `ans_mean`: mean over answer-producing positions (primary; "during generation").
- `first`: position producing the first answer token (depends on the prompt only, so
  it cannot leak answer length).
Layer feature = mean over heads -> L features per example. Also normalized entropy
H / log(n_p) with n_p = number of attended positions.

### Experimental Steps
1. Build the 7,500-question set; 2. generate answers; 3. extract entropy + logit stats;
4. judge; 5. analysis E1-E5.

### Baselines / controls
- Chance (0.5); length-only (prompt tokens, answer tokens).
- Logit baselines: mean token negative log-likelihood, max token NLL, mean predictive
  entropy of the next-token distribution, first-token probability.
- Entropy + length; entropy + logit (does attention add anything?).
- Per-head entropy (L*H features, L2-regularized) as an upper bound for entropy-only.

### Evaluation Metrics
- E1: relative difference of means (halluc - factual)/factual for the last-3-layer mean,
  percentile bootstrap 95% CI (10,000 resamples); Cohen's d; Mann-Whitney U. Per-layer
  d with Benjamini-Hochberg FDR across layers. Same after residualizing entropy on
  log prompt length and log answer length (and for normalized entropy).
- E2-E4: AUROC, out-of-fold predictions from 5-fold stratified CV repeated 5 times
  (standardization fit inside folds, default C=1), bootstrap 95% CI on pooled
  out-of-fold scores. AUPRC reported as secondary.
- E3: within-domain CV AUROC per domain with CI; bootstrap test of
  AUROC(biography) - AUROC(science); 5x5 train-domain -> test-domain transfer matrix.
- alpha = 0.05.

## Expected Outcomes
Support: +15-30% late-layer entropy, pooled AUROC CI above 0.75, biography top /
science bottom in both models. Refutation: effect of other size/sign, AUROC CI below
0.75, or a different ordering. A likely partial outcome is a real but small entropy
shift with moderate AUROC that is partly explained by length.

## Timeline
Setup/data 30 min; generation+extraction 45 min; judging 20 min; analysis 60 min;
report 40 min.

## Potential Challenges
- Disk nearly full on /data (6 GB): venv uses symlinks to the uv cache; only cached
  models; outputs kept small (per-example L x H float16 arrays).
- Entropy/length confound: handled by normalized and residualized variants and the
  length-only baseline.
- Class imbalance differs by domain (PopQA is long-tail, mostly wrong): AUROC is
  prevalence-free; base rates reported.
- Judge errors: quantify agreement with string match; rerun headline numbers on
  string-match labels.

## Success Criteria
All three sub-hypotheses receive a clear supported / not supported / mixed verdict
with CIs, on both models, with the controls above.

## Deviations from plan
1. **Precision.** fp16 produced NaN attention rows with left-padded batches on Qwen2.5;
   both subject models were run in float32 instead (batch 16, length-adaptive).
2. **Judge model.** The planned judge (Qwen2.5-14B-Instruct, 28 GB) could not be kept in
   memory because the four GPUs were shared with other jobs. A 4-bit version ran at
   ~1 s/item (4+ hours) and was stopped after 128 verdicts. The judge actually used is
   Qwen3-8B (bf16, thinking disabled), which is independent of both subject models. On
   the 128 overlapping items the two judges agreed on 123.
3. **Primary label.** Auditing judge-vs-string-match disagreements showed the 8B judge
   errs in both directions (e.g. grading an answer INCORRECT although it is verbatim in
   the gold list; grading "N/A" CORRECT). The primary label was therefore changed to a
   hybrid: correct if a gold alias string-matches OR the judge says CORRECT, with a
   regex abstention rule on top. This was decided from the label audit, after I had seen
   provisional string-match-only results but before any judge-based result. All headline
   numbers are also reported under judge-only and string-match-only labels.
4. **Added controls.** A domain-only classifier, entropy+domain, and "within-domain
   AUROC of pooled scores" were added after the provisional run showed that domain
   identity alone predicts correctness well (pooled AUROC is confounded by base rates).
5. **Analysis speed.** Bootstraps were vectorised; AUROC CIs use 2,000 resamples.
