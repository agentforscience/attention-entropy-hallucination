# Attention Entropy as a Predictor of Hallucination

Tests whether self-attention entropy during answer generation predicts factual errors in
closed-book QA, using Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct on 7,500 questions
across five domains (biography, geography, creative works, science, trivia).
Full write-up: [REPORT.md](REPORT.md).

## Key findings

- **Entropy is not higher when hallucinating.** Final-3-layer attention entropy is lower
  for wrong answers: -18% / -15% pooled, and -2% / -6% after controlling for domain and
  length (Qwen / Llama). The hypothesized +15-30% is not observed anywhere.
- **Pooled AUROC of a layer-wise-entropy classifier is 0.83 / 0.82, but this is inflated.**
  A domain indicator alone scores 0.79 / 0.72. Within domains, entropy reaches 0.62-0.89
  (mean 0.75 / 0.77).
- **Science is the hardest domain (0.62 / 0.69); biography is third of five (0.75 / 0.74),**
  not the best. Creative works and geography are best.
- **A logit baseline is stronger** (0.88 pooled in both models); adding entropy to it
  gains about +0.01 AUROC.
- Per-head entropy (0.89 / 0.87) is much better than per-layer averages.

## Reproduce

```bash
uv venv && source .venv/bin/activate && uv sync      # or: uv pip install -r requirements.txt
# download raw data: see datasets/README.md
python src/prepare_data.py
python src/extract.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen25_7b --batch_size 16
python src/extract.py --model unsloth/Meta-Llama-3.1-8B-Instruct --tag llama31_8b --batch_size 16
python src/judge.py --tags qwen25_7b llama31_8b
for t in qwen25_7b llama31_8b; do
  python src/analyze.py --tag $t
  python src/analyze.py --tag $t --label judge --fast
  python src/analyze.py --tag $t --label strmatch --fast
done
python src/make_figures.py && python src/summarize.py > results/summary_tables.md
```

Needs one GPU with about 40 GB for float32 extraction (about 25 min per model) and about
20 GB for the judge. The analysis is deterministic (seed 42) and runs on CPU in about
2 minutes per model.

## Files

| Path | Purpose |
|---|---|
| `planning.md` | pre-specified plan, plus deviations |
| `REPORT.md` | full report |
| `src/prepare_data.py` | builds the 5-domain question set |
| `src/extract.py` | generation + attention-entropy and logit features |
| `src/judge.py` | LLM-judge and string-match labels |
| `src/analyze.py` | effect sizes, classifiers, per-domain and transfer analyses |
| `src/make_figures.py`, `src/summarize.py` | figures and tables |
| `results/`, `figures/`, `logs/` | outputs |
| `resources.md`, `datasets/README.md` | data and model provenance |

Note: in this workspace `.venv` was populated with `UV_LINK_MODE=symlink` (the data disk
was nearly full), so its packages point into the local uv cache; recreate it with the
commands above on another machine.
