---
name: collect-kaggle-solutions
description: Collect ranked Kaggle competition solution write-ups through a user-specified final rank using Kaggle CLI, synthesize them into a Japanese pipeline-oriented tutorial article for an Expert pursuing Competition Gold, and render one original-language and one Japanese PDF per solution post. Use when the user supplies a Kaggle competition slug or URL and asks to collect, study, summarize, translate, archive, or PDF top-ranked or Competition Gold solutions, including substantive author Q&A while excluding congratulatory comments.
---

# Collect Kaggle Solutions

Run the complete workflow from a Kaggle competition slug or URL and a positive `max_rank`. Work from the project root and store outputs under `solutions/<YYYYMM>-<competition-slug>/`, using the competition end month returned in the Kaggle CLI `deadline` for `YYYYMM`.

The standard Markdown deliverable is `article.md`: an original, cross-solution tutorial organized by the competition's actual problem-solving pipeline. Do not create `summary.md`, rank-by-rank solution cards, or a generic audit report.

## Requirements

- Require `uv`, the project environment, a working Kaggle CLI login, and network access to Kaggle.
- Use Kaggle CLI as the only competition discovery and content acquisition source.
- Do not use web search to fill missing solution posts.
- Translate directly as the running agent. Do not call an external translation API.
- Preserve existing artifacts unless the user explicitly requests refresh or overwrite.
- Require a valid competition `deadline` from Kaggle CLI. Stop for user input rather than guessing the directory prefix when it is absent or invalid.

## Read the policies

Read these files before selecting or writing content:

- [content-selection.md](references/content-selection.md): rank, post, and comment selection.
- [article-format.md](references/article-format.md): evidence worksheet, article structure, writing rules, and quality gate.
- [manifest-schema.md](references/manifest-schema.md): rank-to-topic mapping contract.
- [output-layout.md](references/output-layout.md): final and ignored intermediate files.

## Run the workflow

Set the script path once:

```bash
PIPELINE=.claude/skills/collect-kaggle-solutions/scripts/pipeline.py
```

### 1. Inspect resumable state

```bash
uv run python "$PIPELINE" status <competition>
```

Continue from valid existing files. Do not restart completed work without a reason.

### 2. Collect official data

```bash
uv run python "$PIPELINE" collect <competition> --max-rank <max_rank>
```

This collects competition metadata, pages, the official final leaderboard, and every Discussion-list page exposed by Kaggle CLI. It creates the solution directory only after deriving `YYYYMM` from the competition deadline. Stop for user input if the competition, deadline, authentication, or leaderboard cannot be established.

The collector validates leaderboard rows before assigning ranks. It quarantines rows whose team name looks like a submission filename, saves every retrieved CLI row to `.work/leaderboard-raw.json`, records exclusions in `.work/leaderboard-anomalies.json`, fetches replacement rows, and only then assigns ranks `1..max_rank`. If any anomaly is reported, review both files and confirm that `.work/leaderboard.json` starts with the actual winning team before continuing.

### 3. Build the solution manifest

```bash
uv run python "$PIPELINE" init-manifest <competition>
```

Review `.work/leaderboard.json` and every item in `.work/topics.json`. Match first-party solution posts according to `content-selection.md`, then edit `.work/manifest.json` according to `manifest-schema.md`.

- Include all complementary posts from a ranked team.
- Mark unresolved ranks `not_found` and continue.
- Do not infer missing solutions from third-party posts.

Validate the edited manifest and fix every reported failure before continuing:

```bash
uv run python "$PIPELINE" check-manifest <competition>
```

This check also compares every manifest rank, team name, and private score with the validated leaderboard. Do not rename PDFs or translations to work around a mismatch; correct the leaderboard/manifest mapping and regenerate affected artifacts.

### 4. Fetch selected posts and all comments

```bash
uv run python "$PIPELINE" fetch <competition>
```

Review each `.work/raw/<topic-id>.json`. Select only substantive question-and-author-answer chains according to `content-selection.md`. Add all retained message IDs to the discussion's `selected_comment_ids` in chronological conversational order. Include both question and response IDs.

### 5. Build the evidence worksheet and write the Japanese article

Follow `article-format.md` in order.

1. Review every selected main post and retained author Q&A.
2. Create or update `.work/article-evidence.md` using the evidence worksheet in `article-format.md`. Record each team's thesis, pipeline, strongest reported effect, failures, validation, and constraints before drafting prose.
3. Derive a competition-native pipeline and one central thesis from the completed worksheet. Organize findings by problem and causal stage, not by final rank.
4. Create `solutions/<YYYYMM>-<competition-slug>/article.md` using only collected competition pages, validated leaderboard data, selected main posts, and retained author Q&A as evidence.
5. Run both the manual quality gate in `article-format.md` and the automated `verify` command before reporting completion.

Write for a Competition Expert pursuing Gold, Master, or Grandmaster, but assume the reader did not participate. Explain the task first, compare teams at the point where their approaches diverge, preserve reported numbers and conditions, and make failures and transferable decision rules easy to find. Link source-backed claims inline to their Kaggle Discussions. Never invent a missing team's method or an ablation that the authors did not report.

### 6. Translate each selected discussion

Run `status` again to obtain the exact expected translation paths. Create every listed Japanese Markdown file under `.work/translations/`.

- Translate the complete main post faithfully rather than summarizing it.
- Append `## 補足Q&A` only when `selected_comment_ids` is non-empty.
- Translate exactly the selected Q&A chain and identify each speaker.
- Preserve images, links, tables, code, formulas, identifiers, model names, and numerical values.
- Process and save one discussion at a time so interrupted runs can resume.

### 7. Render the PDFs

```bash
uv run python "$PIPELINE" render <competition> --language all
```

The English/original PDF uses the complete source post plus selected original Q&A. The Japanese PDF uses the corresponding translation Markdown. Continue when an image download fails; retain its source URL when possible.

### 8. Verify all final artifacts

```bash
uv run python "$PIPELINE" verify <competition>
```

Do not report completion until verification passes. Confirm:

- The manifest passes all consistency checks.
- `.work/article-evidence.md` covers every found team and contains a cross-team matrix.
- `article.md` passes the structural and source-coverage checks.
- Each selected Discussion has one original and one Japanese PDF.
- PDFs have pages and extractable text.
- Japanese PDFs contain Japanese text.
- The verification report contains no failures.

## Completion report

Report the number of targeted ranks, found teams, solution posts, original PDFs, Japanese PDFs, and qualifying Q&A chains. List unresolved ranks without implying that no solution exists. Link the final article and both PDF directories.

## Troubleshooting

### A filename appears as a leaderboard team

Treat it as contaminated leaderboard data, not as a real final rank. Review `.work/leaderboard-raw.json` and `.work/leaderboard-anomalies.json`. Continue only when `leaderboard.json` contains exactly ranks `1..max_rank` and rank 1 is the actual winning team. If the row is not detected automatically, stop before `init-manifest`, document the evidence, and extend `leaderboard_anomaly_reasons()` conservatively.

### Existing PDF filenames disagree with the validated leaderboard

Do not only rename the PDFs. Correct the manifest, translations, article, and all rank-bearing PDF contents, regenerate the PDFs with `--overwrite`, and run `verify`.

### The deadline is missing or conflicts with an existing directory

Do not guess the end month or create a second directory for the same competition. Confirm the Kaggle CLI metadata and stop for user input before renaming or overwriting existing artifacts.

## Examples

- `uv run python "$PIPELINE" collect rsna-intracranial-aneurysm-detection --max-rank 11` quarantines a row such as `perfect_submission.parquet`, obtains the next valid team, and writes validated ranks 1 through 11.
- `uv run python "$PIPELINE" check-manifest <competition>` fails when a manifest team or score is shifted relative to `leaderboard.json`.
- A medical-imaging article may use `DICOM normalization → candidate localization → local classification → series aggregation → runtime` as its body, while an audio article may use `domain adaptation → representation → event modeling → long context → blending → CPU inference`. Derive sections from the evidence; do not reuse either sequence mechanically.
