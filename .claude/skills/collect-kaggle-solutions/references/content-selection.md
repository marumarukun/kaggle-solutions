# Content selection policy

## Rank scope

- Use the official final leaderboard.
- Before assigning ranks, reject CLI rows whose `teamName` is empty or looks like a submission artifact filename (for example `perfect_submission.parquet`). Preserve rejected rows and their reasons in `.work/leaderboard-anomalies.json`, fetch enough additional rows to fill the requested scope, then assign contiguous ranks.
- Review `.work/leaderboard-raw.json`, `.work/leaderboard-anomalies.json`, and the sanitized `.work/leaderboard.json` whenever an anomaly is detected. Do not silently trust the CLI row order after contamination is observed.
- Treat every final rank from 1 through `max_rank` as an acquisition candidate.
- Do not infer the Competition Gold cutoff. The user-supplied `max_rank` is the scope boundary.
- Keep all ranks in the acquisition report even when no solution post is found.

## Acquisition sources

- Use Kaggle CLI for competition pages, the final leaderboard, discussion listings, main posts, and comments.
- Search every discussion-list page exposed by Kaggle CLI before marking a rank as not found.
- Do not use general web search, Kaggle site search, search-engine snippets, or an external dataset to discover additional posts.
- When Kaggle CLI does not expose a matching solution post, record it as not found and continue. Do not attempt to fill the gap by inference.

## Solution post matching

Match candidate posts against final rank, team name, author identity, title, and body content. A title containing a rank is evidence but is not sufficient by itself when it conflicts with the final leaderboard.

Include:

- First-party posts that explain a ranked team's solution.
- Multiple complementary posts from the same team.
- A teammate's post when authorship or team membership can be established.

Exclude:

- Posts that merely congratulate, ask where a solution is, or link to unrelated material.
- Speculative reconstructions written by people outside the team.
- Generic competition retrospectives that do not explain the team's method.

Record the evidence used for each rank-to-post match in `manifest.json`.

## Main post and comment handling

Always retain the complete main solution post, including its images, tables, code blocks, and links.

Inspect all comments and retain a comment thread only when:

1. A participant asks a substantive question about the solution, implementation, data, validation, feature engineering, loss, ensemble, inference, or reported result.
2. The solution author or a confirmed teammate responds.
3. The response adds, corrects, or clarifies information useful for understanding or reproducing the solution.

When retained, include the minimum complete conversational chain: question, author response, and any technical follow-up needed to understand that response.

Exclude:

- Congratulations, thanks, emojis, and social conversation.
- Questions with no answer from the author or a confirmed teammate.
- Repetition of information already clear in the main post unless the answer resolves ambiguity.
- Unconfirmed third-party advice. It may be retained only when the author explicitly confirms or expands on it and that exchange adds technical information.

Append retained exchanges under `Supplemental Q&A` in each PDF. Preserve the original order and identify each speaker.

## English and Japanese PDF content

- English PDF: complete original main post plus selected original-language Q&A.
- Japanese PDF: a faithful Japanese translation of the same selected content.
- Do not replace a PDF translation with a summary.
- Preserve code, URLs, model names, identifiers, formulas, tables, and numerical values.
- Perform translation directly as the running agent. Do not call an external translation API.

## Article use of Q&A

Integrate a clarification into `article.md` only when it materially changes or improves understanding of the method. Do not reproduce every selected Q&A exchange in the article.

## Stop conditions

Stop and request user input when:

- The competition cannot be identified from the slug or URL.
- Kaggle authentication is unavailable.
- The official final leaderboard cannot be obtained, so ranks cannot be established.
- A suspicious leaderboard row cannot be confidently classified as either a ranked team or a submission artifact.
- Multiple plausible competitions remain after normalization.

Continue and report the limitation when:

- A rank has no discovered solution post.
- A team has multiple solution posts.
- A post has no qualifying Q&A.
- An image cannot be downloaded. Preserve its source URL when possible.
