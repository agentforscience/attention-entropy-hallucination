"""Build the 5-domain closed-book QA set (1,500 questions per domain, seed 42).

Output: datasets/processed/questions.jsonl with fields
  qid, domain, source, relation, question, gold (list of acceptable answers)
"""
import json
import os
import re

import numpy as np
import pandas as pd

SEED = 42
N_PER_DOMAIN = 1500
RAW = "datasets/raw"
OUT = "datasets/processed/questions.jsonl"

POPQA_DOMAINS = {
    "biography": ["occupation", "place of birth", "father", "mother"],
    "geography": ["country", "capital", "capital of"],
    "creative_works": ["author", "director", "producer", "screenwriter", "composer"],
}


def main():
    rng = np.random.RandomState(SEED)
    rows = []

    pop = pd.read_csv(f"{RAW}/PopQA/test.tsv", sep="\t")
    for domain, props in POPQA_DOMAINS.items():
        sub = pop[pop.prop.isin(props)]
        assert len(sub) >= N_PER_DOMAIN, (domain, len(sub))
        sub = sub.sample(N_PER_DOMAIN, random_state=rng)
        for r in sub.itertuples():
            rows.append(dict(qid=f"popqa-{r.id}", domain=domain, source="PopQA", relation=r.prop,
                             question=r.question, gold=json.loads(r.possible_answers)))

    sci = pd.read_parquet(f"{RAW}/sciq/data/train-00000-of-00001.parquet")
    # Keep self-contained questions: must end with '?' (drops cloze-style fragments).
    sci = sci[sci.question.str.strip().str.endswith("?")].drop_duplicates("question")
    sci = sci.sample(N_PER_DOMAIN, random_state=rng)
    for i, r in zip(sci.index, sci.itertuples()):
        rows.append(dict(qid=f"sciq-{i}", domain="science", source="SciQ", relation="",
                         question=r.question.strip(), gold=[r.correct_answer.strip()]))

    tq = pd.read_parquet(f"{RAW}/trivia_qa/rc.nocontext/validation-00000-of-00001.parquet")
    tq = tq.drop_duplicates("question").sample(N_PER_DOMAIN, random_state=rng)
    for r in tq.itertuples():
        gold = list(dict.fromkeys([r.answer["value"], *r.answer["aliases"]]))
        rows.append(dict(qid=f"tqa-{r.question_id}", domain="trivia", source="TriviaQA", relation="",
                         question=re.sub(r"\s+", " ", r.question).strip(), gold=gold))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    df = pd.DataFrame(rows)
    assert df.qid.is_unique and df.question.str.len().min() > 5
    print(df.groupby(["domain", "source"]).size())
    print(df.groupby("domain").relation.value_counts().to_string())
    print("question chars:", df.question.str.len().describe().round(1).to_dict())
    print(df.groupby("domain").head(2)[["domain", "question", "gold"]].to_string())


if __name__ == "__main__":
    main()
