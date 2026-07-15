# Summary format

Create `solutions/<YYYYMM>-<competition-slug>/summary.md` in Japanese. Derive `YYYYMM` from the competition end date in the Kaggle CLI `deadline`. Use the following order.

## 1. Title and metadata

- Competition title and a phrase indicating that this is a pre-reading guide to top solutions.
- Target final ranks: 1 through `max_rank`.
- Competition start and end dates. Include the deadline timezone when available.
- Retrieval date.
- Purpose: understand the whole field, shared ideas, and differences before reading each write-up closely.
- Retrieval method: Kaggle CLI and the data sources actually used.

## 2. Key conclusions

State approximately five to eight conclusions that explain what decided the competition. Prefer causal technical insights over a list of model names.

## 3. Competition-specific glossary

Place this near the beginning so that the rest of the document is easy to read. Define:

- Dataset-specific entities, columns, labels, metrics, and evaluation units.
- Competition-specific abbreviations and domain terms.
- Terms whose meaning differs from ordinary machine-learning usage.

Do not turn this into a general machine-learning glossary.

## 4. Frequently used technology stack

Start with a comparison table containing:

| Technology | Ranks using it | Short description | Value in this competition |
|---|---|---|---|

Then explain the important technologies in short subsections. For each technology, cover:

1. What it is.
2. Why it matched this competition's data or metric.
3. Which solution components benefited from it.

## 5. Competition overview

Summarize the task, inputs, outputs, metric, execution constraints, and the main technical difficulties.

## 6. Acquisition status

List every final rank from 1 through `max_rank`, including ranks with no discovered write-up.

| Final rank | Team | Private score | Solution discussion | Status |
|---:|---|---:|---|---|

If one team published multiple complementary solution posts, list all of them.

## 7. Solution map

Provide a compact comparison table using dimensions that matter for the competition, such as main model, representation, validation, temporal modeling, ensembling, and post-processing.

## 8. Individual solution summaries

Write one section per discovered team in final-rank order. When a team has multiple posts, combine the team's overall story while distinguishing each author's contribution.

Each section should be understandable in one to two minutes and contain:

- A one-sentence characterization.
- The central pipeline and modeling decisions.
- Important preprocessing, validation, loss, ensemble, and post-processing choices.
- Quantitative results when the source gives them.
- Material clarifications found in selected author Q&A.
- Why the approach created value in this competition.

## 9. Cross-solution comparison

Compare common patterns, meaningful differences, and rank-dependent trends. Separate source-backed facts from synthesis or inference.

## Writing rules

- Optimize for comprehension before close reading, not for replacing the original write-ups.
- Preserve model names, code identifiers, column names, metrics, and numbers accurately.
- Link every discovered solution post.
- Do not include a generic cautions section.
- Do not include questions for the reader to answer while close-reading.
- Avoid repeating the same explanation in the glossary, technology section, and individual summaries.
- Do not claim that an undiscovered write-up does not exist; state only that it was not found in the searched sources.
