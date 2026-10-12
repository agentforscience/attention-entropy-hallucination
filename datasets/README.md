# Datasets

Raw files are git-ignored (see `.gitignore`); re-download with:

```python
from huggingface_hub import hf_hub_download
for repo, f in [("akariasai/PopQA", "test.tsv"),
                ("allenai/sciq", "data/train-00000-of-00001.parquet"),
                ("mandarjoshi/trivia_qa", "rc.nocontext/validation-00000-of-00001.parquet")]:
    hf_hub_download(repo, f, repo_type="dataset", local_dir=f"datasets/raw/{repo.split('/')[1]}")
```

Then `python src/prepare_data.py` writes `datasets/processed/questions.jsonl`
(7,500 questions, 1,500 per domain, seed 42).

| Domain | Source | Selection |
|---|---|---|
| biography | PopQA | relations: occupation, place of birth, father, mother |
| geography | PopQA | relations: country, capital, capital of |
| creative_works | PopQA | relations: author, director, producer, screenwriter, composer |
| science | SciQ train | questions ending in "?", deduplicated; gold = `correct_answer` |
| trivia | TriviaQA rc.nocontext validation | deduplicated; gold = value + aliases |
