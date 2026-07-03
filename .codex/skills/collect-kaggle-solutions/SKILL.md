---
name: collect-kaggle-solutions
description: Collect ranked Kaggle competition solution write-ups through a user-specified final rank using Kaggle CLI, create a Japanese pre-reading summary, and render one original-language and one Japanese PDF per solution post. Use when the user supplies a Kaggle competition slug or URL and asks to acquire, summarize, translate, archive, or PDF the top-ranked or Competition Gold solutions, including substantive author Q&A while excluding congratulatory comments.
---

# Collect Kaggle Solutions

Run the complete workflow from a Kaggle competition slug or URL and a positive `max_rank`. Work from the project root and store outputs under `outputs/<competition-slug>/`.

## Requirements

- Require `uv`, the project environment, a working Kaggle CLI login, and network access to Kaggle.
- Use Kaggle CLI as the only competition discovery and content acquisition source.
- Do not use web search to fill missing solution posts.
- Translate directly with Codex. Do not call an external translation API.
- Preserve existing artifacts unless the user explicitly requests refresh or overwrite.

## Read the policies

Read these files before selecting or writing content:

- [content-selection.md](references/content-selection.md): rank, post, and comment selection.
- [summary-format.md](references/summary-format.md): required `summary.md` structure.
- [manifest-schema.md](references/manifest-schema.md): rank-to-topic mapping contract.
- [output-layout.md](references/output-layout.md): final and ignored intermediate files.

## Run the workflow

Set the script path once:

```bash
PIPELINE=.codex/skills/collect-kaggle-solutions/scripts/pipeline.py
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

This collects competition metadata, pages, the official final leaderboard, and every Discussion-list page exposed by Kaggle CLI. Stop for user input if the competition, authentication, or leaderboard cannot be established.

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

### 4. Fetch selected posts and all comments

```bash
uv run python "$PIPELINE" fetch <competition>
```

Review each `.work/raw/<topic-id>.json`. Select only substantive question-and-author-answer chains according to `content-selection.md`. Add all retained message IDs to the discussion's `selected_comment_ids` in chronological conversational order. Include both question and response IDs.

### 5. Write the Japanese pre-reading summary

Create `outputs/<competition-slug>/summary.md` according to `summary-format.md`. Use only collected competition pages, leaderboard data, selected main posts, and retained author Q&A as evidence.

Write for understanding before close reading. Explain competition-specific vocabulary early and describe frequently used technology both generally and in terms of the value it created in this competition.

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
- `summary.md` exists.
- Each selected Discussion has one original and one Japanese PDF.
- PDFs have pages and extractable text.
- Japanese PDFs contain Japanese text.
- The verification report contains no failures.

## Completion report

Report the number of targeted ranks, found teams, solution posts, original PDFs, Japanese PDFs, and qualifying Q&A chains. List unresolved ranks without implying that no solution exists. Link the final summary and both PDF directories.
