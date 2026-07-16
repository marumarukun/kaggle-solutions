# Pipeline tutorial article format

Create `solutions/<YYYYMM>-<competition-slug>/article.md` in Japanese. Derive `YYYYMM` from the Kaggle CLI competition `deadline`.

The reader is a Kaggle Competition Expert pursuing Gold, Master, or Grandmaster, but may not have entered this competition. Teach how the task was decomposed, which interventions mattered, when they failed, and what decision process transfers elsewhere. Produce an original tutorial article; do not imitate a named writer's distinctive wording.

## Non-negotiable outcome

The article must let the reader answer these questions without opening every source first:

1. What was predicted, from what input, under which metric and execution constraints?
2. What was the central bottleneck shared by top teams?
3. What causal pipeline did strong solutions build around that bottleneck?
4. Where did teams make meaningfully different choices, and what evidence supports them?
5. Which attractive ideas failed, under what conditions, and what rule should transfer?

Organize the main body by the competition's causal pipeline, not by rank. Mention ranks only as evidence for a technique or counterexample.

## Step 1: Build the evidence worksheet

Before drafting `article.md`, create `.work/article-evidence.md`. It is an intermediate reasoning artifact and must not be the final deliverable.

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
- Candidate article stages: <one or more causal stages>
```

Then add `## Cross-team matrix` and this table:

| Candidate stage or bottleneck | Teams / ranks | Shared pattern | Important exception | Strongest evidence | Transferable rule |
|---|---|---|---|---|---|

Evidence worksheet rules:

- Review every selected post before choosing the article thesis. Do not let the first-place solution define the whole narrative by default.
- Copy model names, identifiers, scores, metric axes, and before/after directions exactly.
- Record “not reported” instead of manufacturing an ablation.
- Treat a team's reported delta as evidence under that team's conditions, not as a controlled cross-team comparison.
- Record negative results and runtime or selection failures with the same care as successful methods.
- Use retained author Q&A only when it clarifies implementation, evidence, or transfer conditions.
- Keep source URLs beside evidence while extracting it. This prevents unsupported prose during drafting.

Do not start the article until every found team has a block and the cross-team matrix is complete.

## Step 2: Derive the story from the evidence

### Select the central thesis

Choose one sentence that explains the competition-specific bottleneck and connects several independently reported top solutions. A good thesis describes a problem and the strategic response, for example:

> When the useful signal is tiny relative to the input, narrow the search space before classification and preserve local evidence until aggregation.

Reject theses that merely say ensembling, preprocessing, or strong models were important. The thesis should explain why the winning pipeline took its shape.

### Build a causal pipeline

Cluster the cross-team matrix into approximately five to eight stages. Order them by dependency:

1. data generation or domain gap;
2. preprocessing, localization, or representation;
3. core prediction;
4. context, aggregation, or specialization;
5. validation and selection;
6. ensemble, post-processing, or inference constraints.

These are prompts, not mandatory headings. Use only stages that fit the competition. Merge adjacent stages when they solve the same problem, and split a stage when the top teams made an important, teachable choice there.

### Select evidence for the article

For each stage, select:

- at least two teams when a cross-team pattern exists;
- the strongest quantitative result available;
- an exception or failure that limits the claim;
- the reason the technique matched the task, metric, data, or runtime.

Every discovered team must appear in at least one substantive body claim unless its post contains no usable technical detail. Every discovered post must still appear in the reference list.

## Step 3: Write `article.md`

Use the following fixed outer structure. Only the numbered pipeline sections are competition-adaptive.

### Title

```markdown
# <Competition title> 上位解法まとめ — <central thesis in plain Japanese>
```

The subtitle must express the competition-specific lesson. Avoid generic subtitles such as “Gold solutions explained.”

### `## はじめに`

In approximately four to seven short paragraphs:

- introduce the real task and link the competition page;
- state when the competition ended;
- contrast the diversity of model families with the common strategic pattern;
- place the central thesis in a short emphasized blockquote;
- state the target rank range and number of found teams;
- identify unresolved ranks and say only that no first-party Solution was found in the Kaggle CLI-exposed Discussions searched;
- mention a validated leaderboard anomaly only when one occurred.

Do not start with collection mechanics or a metadata table.

### `## コンペ概要`

Use these subsections:

1. `### タスク`: explain input, output granularity, metric, and the metric's strategic consequence in plain language.
2. `### 提供データ`: use one compact table with `データ`, `内容`, and `解法上の役割`.
3. `### このコンペが難しい理由`: give approximately four to seven concrete bullets tied to the data, metric, validation, or execution environment.

Assume the reader did not participate. Define competition-native abbreviations before relying on them.

### `## 上位解法の全体像`

Draw one original Mermaid flowchart that shows the full winning pipeline from input and supervision to prediction. Follow it with approximately three to six numbered axes explaining where teams differed.

The diagram must synthesize the collected evidence. Do not copy a source diagram or decorate the article with an unrelated architecture chart.

### Adaptive numbered pipeline sections

Create approximately five to eight `## <number>. <stage>` sections. Name each stage after the problem being solved, not a model family. Prefer “探索範囲を絞る” over “nnU-Net,” and “録音環境の差を埋める” over “EfficientNet.”

Within each stage, move through this teaching sequence when evidence permits:

1. explain the failure mode or task constraint;
2. compare how multiple teams addressed it;
3. show the strongest reported quantitative or precise qualitative evidence;
4. explain the mechanism in task-specific terms;
5. show an exception, failed variant, or transfer condition;
6. state the decision rule the reader should retain.

Use short descriptive `###` subsections rather than repeating team cards. A paragraph may compare several teams; link each source-backed claim inline to the relevant Kaggle Discussion.

Add at least one more original Mermaid diagram beyond the overview diagram. Use it for the most important mechanism, branching choice, or trade-off. Additional visualizations may use the smallest useful form:

- Mermaid flowchart for a mechanism or alternative pipeline;
- table for repeated-field comparison or ablation;
- compact text diagram for a simple transformation.

Aim for two to four diagrams total, with at least two in Mermaid. Each must explain a relationship that prose alone would make harder to grasp.

### `## 上位解法から見えた、特に重要な発見`

Write approximately four to six numbered `###` findings. Each finding must combine evidence from earlier sections into one memorable causal conclusion. Do not introduce unsupported new facts or merely repeat model names.

### `## うまくいかなかったアプローチ`

Select approximately six to ten high-signal failures. Format each bullet as:

```markdown
- **<tempting approach>**: <what happened, under which condition, and what decision rule follows>
```

Prefer counterexamples involving domain shift, excessive pseudo-label rounds, harmful resampling, validation shake-up, ensemble correlation, hard filtering, or runtime failure when the sources support them. End with a short synthesis explaining why the failures occurred; do not imply that a method is universally bad.

### `## まとめ`

First reconstruct the winning pipeline as approximately five to eight numbered steps. Then state the transferable principle in an emphasized blockquote. Finish with a short paragraph naming other task families where the reasoning may apply and the condition that makes the transfer valid.

Do not write a “next time in this competition” experiment list. The same competition may not recur.

### `## 参照した上位Solution`

List every discovered first-party Solution in final-rank order. Preserve complementary posts as separate list items when needed. Include rank, recognizable title or team/author, and the direct Kaggle Discussion URL.

## Source and accuracy rules

- Use only collected competition pages, validated leaderboard data, selected first-party posts, and retained author Q&A.
- Link factual method claims and all reported results inline, near the relevant sentence. The final reference list alone is not sufficient attribution.
- Preserve the metric axis. Distinguish CV, Public LB, Private LB, single-fold, single-model, and ensemble results.
- Write “the team reported” or equivalent when causality rests on one team's experiment.
- When synthesizing across teams, use language such as “the collected Solutions suggest” and make the inference traceable to preceding evidence.
- Never infer a missing rank's method, team authorship, or unreported experiment.
- Do not treat leaderboard differences between teams as ablations.
- Preserve technical identifiers in their conventional form; otherwise prefer natural Japanese over unnecessary English fragments.
- Explain an important term once, then use it consistently.

## Human-readable writing rules

- Teach the problem before the catalog of methods.
- Keep paragraphs focused on one causal point and lead with the conclusion.
- Use concrete numbers when they change prioritization; omit low-value implementation trivia.
- Give each table and diagram one job and explain the takeaway immediately before or after it.
- Compare teams inside the relevant pipeline stage so the reader sees alternatives at the moment of decision.
- Avoid exhaustive team cards, acquisition-status tables, evidence labels on every sentence, audit ledgers, and generic caution sections.
- Avoid repeating the same fact in the introduction, body, findings, failures, and conclusion unless the later occurrence adds a new decision rule.
- Keep the article skimmable with descriptive headings, but make the prose coherent when read continuously.

## Manual quality gate

Before running automated verification, confirm all of the following:

- [ ] `.work/article-evidence.md` contains every found team and a completed cross-team matrix.
- [ ] A non-participant can explain the input, target, evaluation unit, metric, constraints, and central difficulty from the opening sections.
- [ ] The title and opening blockquote express a competition-specific causal thesis.
- [ ] Main sections follow the actual problem-solving pipeline rather than rank order or a reusable generic taxonomy.
- [ ] Every found team contributes at least one substantive body claim when usable technical detail exists.
- [ ] Every discovered post is linked in the final reference list and source-backed claims use inline links.
- [ ] The article includes at least two original Mermaid diagrams, including the overall pipeline.
- [ ] Reported numbers preserve their metric axis, conditions, direction, and source wording.
- [ ] At least one exception or failure constrains each major conclusion where evidence permits.
- [ ] The important findings are causal syntheses rather than architecture counts.
- [ ] The failure section contains memorable decision rules without becoming an exhaustive experiment ledger.
- [ ] The conclusion transfers a decision process, not a list of competition-specific model names.
- [ ] Missing ranks are disclosed without inferred methods or claims that no Solution exists.
- [ ] Repeated facts, generic ML explanations, and low-value implementation details have been removed.
- [ ] Mermaid fences are balanced and diagrams use valid, readable syntax.
- [ ] `article.md` is the only standard synthesized Markdown deliverable; no new `summary.md` or “refined” variant was created.
