# Resources

The resource-finding stage was skipped for this run (`.neurico/pipeline_state.json`:
`resource_finder.outputs.skipped = true`), so no `literature_review.md`, `papers/` or
`code/` were provided. Everything below was gathered during the experiment session.

## Datasets (downloaded from the Hugging Face Hub, see `datasets/README.md`)
- PopQA (`akariasai/PopQA`, test.tsv, 14,267 entity-centric questions with Wikidata relation labels)
- SciQ (`allenai/sciq`, train split, 11,679 science exam questions)
- TriviaQA (`mandarjoshi/trivia_qa`, rc.nocontext validation, 17,944 questions)

## Models (already in the local HF cache; no API models used)
- Subjects: `Qwen/Qwen2.5-7B-Instruct`, `unsloth/Meta-Llama-3.1-8B-Instruct` (mirror of Meta's weights)
- Judge: `Qwen/Qwen2.5-14B-Instruct`
- No `OPENROUTER_KEY` / `OPENAI_API_KEY` was set in the environment. The study needs
  attention maps, which hosted APIs do not expose, so local models were required for the
  subjects regardless; the missing key only affected the choice of judge.

## Code
All code in `src/` was written for this study; no external repositories were used.

## Prior work referenced (from memory, not downloaded)
See the References section of `REPORT.md`.
