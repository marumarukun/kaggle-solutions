# Concise big-picture article format

Create `solutions/<YYYYMM>-<competition-slug>/article.md` in Japanese. Derive `YYYYMM` from the Kaggle CLI competition `deadline`.

The reader is a Kaggle Competition Expert pursuing Gold, Master, or Grandmaster who did not enter this competition. They want the **big picture**: what decided the competition, why, and a few representative examples. They do not want every team's method. Write so that a non-participant understands the main point in one pass, without opening any source.

Two finished articles are the reference examples for tone, length, and structure:

- `solutions/202609-biohub-cell-tracking-during-development/article.md`
- `solutions/202608-rogii-wellbore-geology-prediction/article.md`

Read one of them before drafting. Do not copy their wording or competition-specific content.

## Non-negotiable outcome

After one read, the reader can answer:

1. What was predicted, from what input, and how was it scored?
2. Which properties of the metric or data were counterintuitive?
3. What three to five decisions separated the top teams, and why did each work?
4. What tempting ideas failed?

Optimize for comprehension, not coverage. If a fact does not help answer one of these questions, leave it in the evidence worksheet.

## Step 1: Build the evidence worksheet

Before drafting `article.md`, create `.work/article-evidence.md`. It is the place for exhaustive detail, so the article does not have to carry it.

Add one block for every discovered team:

```markdown
## Rank <rank> — <team>

- Source posts: <all first-party Discussion URLs>
- One-sentence thesis: <what this solution did differently or especially well>
- Pipeline: <data → preprocessing → representation/model → aggregation → final prediction>
- Strongest reported effects: <exact before/after, delta, or precise qualitative result with metric axis>
- Data and supervision: <external data, pseudo-labels, manual labels, auxiliary targets, sampling>
- Validation and selection: <split, OOF, Public/Private behavior, model-selection lesson>
- Ensemble and post-processing: <only material choices>
- Runtime and implementation constraints: <only material choices>
- Failures and caveats: <failed experiments, limits, conditions>
- Useful author Q&A: <clarification that changes understanding; otherwise none>
- Candidate article stages: <one or more decisive levers>
```

Then add `## Cross-team matrix` and this table:

| Candidate lever or bottleneck | Teams / ranks | Shared pattern | Important exception | Strongest evidence | Transferable rule |
|---|---|---|---|---|---|

Worksheet rules:

- Review every selected post before choosing the thesis. Do not let the first-place solution define the whole narrative by default.
- Copy model names, identifiers, scores, metric axes, and before/after directions exactly.
- Record "not reported" instead of manufacturing an ablation.
- Treat a team's reported delta as evidence under that team's conditions, not as a controlled cross-team comparison.
- Record negative results and runtime or selection failures with the same care as successes.
- Keep source URLs beside evidence.

Do not start the article until every found team has a block and the matrix is complete.

## Step 2: Choose the thesis and the decisive levers

### Central thesis

Write one sentence that names the competition-specific bottleneck and the strategic response shared by several top teams. Reject generic theses such as "ensembling and good preprocessing mattered."

### Select three to five levers ("勝負所")

From the cross-team matrix, pick the three to five levers that best explain the final ranking. Rank candidates by two criteria:

- **Impact**: the largest reported single gain for any team, or a large Public/Private effect.
- **Breadth**: how many found teams used it.

A lever that **more than half of the found teams used**, or that was **the largest single improvement for any team**, must become a main lever. Do not push it into the supplementary table, even if it seems less elegant than the others. Example: in ROGII, synthetic data was used by more than half of the teams and was the largest single gain for one team, so it needed its own section, not a table row.

Everything else that mattered, but less, goes into one supplementary table. Facts that do not fit either stay in the worksheet.

### Representative examples

For each lever, choose **one or two representative teams**, and at most three. Prefer:

- the clearest quantitative before/after;
- the most intuitive mechanism;
- one counterexample or failure, when it sharpens the point.

Do not list every team that used the lever.

## Step 3: Write `article.md`

Target length: **body of about 9,000–12,000 Japanese characters**, measured from the title to just before `## 参照した上位Solution`. `verify` fails above 13,000 characters and checks for three to five numbered lever sections.

Use this fixed outer structure.

### Title

```markdown
# <Competition title> 上位解法まとめ — <central thesis in plain Japanese>
```

### `## はじめに`

Short. In order:

1. One or two sentences: what the competition was, with a link, and when it ended.
2. One sentence: model families varied, but the decisive points were shared. "結論を先に書きます。"
3. The thesis as an emphasized blockquote (`> **...**`), two to three sentences.
4. One sentence previewing the levers.
5. Scope: target ranks, found teams and posts, and unresolved ranks, stating only that no first-party Solution was found in the Kaggle CLI-exposed Discussions searched.

### `## コンペ概要`

- `### タスク`: three bullets: **入力**, **出力**, **評価**. Then "この指標には、直感に反する性質が3つあります。" followed by a numbered list of three counterintuitive properties. Each item starts with a bold one-line claim, then one to three plain sentences, with one representative number if it helps.
- `### 提供データ`: a two-column table (`データ`, `内容`) with four to six rows.
- `### このコンペが難しい理由`: three to five one-line bullets.

Define every abbreviation or domain term at first use. If more than about five need defining, add a short glossary table at the end of `コンペ概要`.

### `## 上位解法の全体像`

1. One sentence describing the shared skeleton, then "差がついたのは、下図で色をつけたN か所です。"
2. A Mermaid flowchart of the shared skeleton, with each lever as a highlighted node (`classDef key fill:#fde68a,stroke:#b45309,color:#000`) linked by a dotted edge to the stage it affects.
3. Optionally, one sentence defining the central optimization or model term.

### Numbered lever sections

Write three to five `## <number>. <lever>` sections. Name each section after the decision, in plain Japanese ("注釈のない細胞を負例にしない", "Publicを信じすぎない"), not after a model family. Mark the most important one with "この節が本記事の中心です。"

Within a section, use these labeled building blocks when they fit:

- `**問題**:` the failure mode in one to three sentences, ideally with a concrete picture.
- `**上位の対処**:` the shared response in one sentence, then one to three bullets, each starting with a bold rank (`- **4位**は…`), each with one key number and an inline link.
- `**逆をやるとどうなるか**:` or a counterexample paragraph.
- `**なぜ効くか**:` the mechanism in plain words, when it is not obvious.

Use `###` subsections only when a lever has distinct steps. Put the second required Mermaid diagram in the central lever section. It should show the key mechanism, ideally as a before/after comparison (for example, uniform cost versus per-instance cost, or "decide immediately" versus "keep candidates and decide globally").

### `### 補足：そのほかに効いた工夫`

Place this at the end of the last lever section. One table:

| 工夫 | なぜ効くか | 代表例 |
| --- | --- | --- |

Use four to seven rows. Each 代表例 cell holds one team, one number, and an inline link. This table is also where you cite teams that are not in the main sections.

### `## 上位解法から見えた、特に重要な発見`

Three or four `###` items, each two to three sentences. Explain **why** the levers worked under this competition's conditions. Do not repeat numbers from the body. Refer to section numbers instead.

### `## うまくいかなかったアプローチ`

Five or six bullets:

```markdown
- **<tempting approach>**: <what happened, one number> <one-line rule>（[N位](url)）。
```

End with one sentence naming the shared cause.

### `## まとめ`

1. "上位解法の流れをN行にまとめると、次のとおりです。" followed by a five to seven step numbered list. Each step is one line.
2. The transferable principle as an emphasized blockquote (`> **...**`).
3. One or two sentences on other task families where the reasoning applies, and the condition that makes the transfer valid.

### `## 参照した上位Solution`

One sentence, then a table in final-rank order:

| 順位 | チーム | 一言で言うと | 投稿 |
| --- | --- | --- | --- |

`一言で言うと` is one plain-Japanese phrase, under about 40 characters. Put all of a team's posts in its row. After the table, list unresolved ranks with their team names.

## Writing rules

- Write in です/ます style, in natural Japanese. Avoid literal English sentence structure and unnecessary English fragments, such as "datum branch" or "coherent posterior", unless the term is defined.
- Lead every paragraph with its conclusion. One paragraph makes one point.
- At most one or two numbers per sentence. If a sentence would hold three teams or more than two numbers, split it into bullets or a table.
- When a cause-and-effect step is not obvious, add one sentence that explains why. Arithmetic examples are good ("+1.0 − 1.2 = −0.2 なので常に損").
- Prefer concrete pictures ("胚全体が1フレームで5 µm流れていれば…") over abstract statements.
- Use one term per concept throughout. Define it once.
- Mention ranks only as evidence for a lever, never as the organizing principle.
- Do not repeat the same fact in the introduction, body, findings, and conclusion.

## Source and accuracy rules

- Use only collected competition pages, validated leaderboard data, selected first-party posts, and retained author Q&A.
- Link every factual method claim and reported result inline, near the sentence. The reference table alone is not sufficient attribution. `verify` requires every found rank to have at least one inline link besides its reference-table link. Satisfy this through the lever sections, the supplementary table, or the failure list; do not add paragraphs only for coverage.
- Preserve the metric axis (CV, Public, Private, single model, ensemble) and the direction of change.
- When simplifying, do not strengthen a claim. Keep qualifiers such as "学習データ上で", "ごく一部の例外を除き", and "複数要素の同時変更".
- Label interpretations that are not stated in the post ("〜と考えられます", "〜と読み取れます").
- Never infer a missing rank's method or an unreported ablation. Do not treat leaderboard differences between teams as ablations.

## Step 4: Independent readability review

After the article passes `verify`, ask an independent reviewer to read it as a non-participant. Use a subagent on a different model when available; otherwise use a fresh subagent. Ask for:

- passages that are hard to understand intuitively, with replacement text;
- overloaded paragraphs;
- undefined terms;
- repetition;
- whether a non-participant can follow the main argument in one read.

The reviewer must not edit files. Check every suggestion against the sources before applying it. Reviewers tend to guess identifiers and add unsupported causal claims. Re-run `verify` after applying.

## Manual quality gate

- [ ] `.work/article-evidence.md` contains every found team and a completed cross-team matrix.
- [ ] A non-participant can state the task, the metric, and its counterintuitive properties after reading `コンペ概要`.
- [ ] The thesis is competition-specific and appears as a blockquote in `はじめに`.
- [ ] There are three to five lever sections, chosen by impact and breadth. No lever used by more than half of the found teams, and no team's largest single gain, is relegated to the supplementary table.
- [ ] Each lever has one to three representative teams, not a full list.
- [ ] The overview Mermaid highlights the levers, and a second Mermaid explains the central mechanism.
- [ ] Every sentence holds at most two numbers. No paragraph compares more than two teams in running prose.
- [ ] Every abbreviation and domain term is defined at first use.
- [ ] `発見` explains why, without repeating body numbers.
- [ ] The reference section is the one-line summary table and lists unresolved ranks.
- [ ] The body is within the length target, and `verify` passes.
- [ ] The independent readability review was run, and its accepted suggestions were fact-checked.
