# Manifest schema

Create the initial manifest with `pipeline.py init-manifest`. Then edit `.work/manifest.json` after reviewing the leaderboard and discussion list. Validate the result with `pipeline.py check-manifest`; `fetch` and `verify` enforce the same rules and refuse an inconsistent manifest.

## Rank entries

Keep exactly one entry for every rank from 1 through `max_rank`.

```json
{
  "rank": 6,
  "team": "example team",
  "private_score": "0.50000",
  "status": "found",
  "topic_ids": [123456, 123457]
}
```

Use only these final status values:

- `found`: at least one first-party solution post was matched.
- `not_found`: no qualifying post was found after reviewing all CLI-listed topics.

## Discussion entries

Create one flat entry per selected solution post. Multiple entries may share a rank and team.

```json
{
  "rank": 6,
  "team": "example team",
  "topic_id": 123456,
  "slug": "author-part",
  "match_evidence": "Title states 6th place and the author identifies the ranked team.",
  "selected_comment_ids": [900001, 900002]
}
```

Required fields:

- `rank`: official final rank.
- `team`: exact leaderboard team name.
- `topic_id`: Kaggle Discussion topic ID.
- `slug`: short ASCII filename component unique within the rank when practical.
- `match_evidence`: concise evidence supporting the rank-to-post match.
- `selected_comment_ids`: IDs of both the substantive question and author response, plus any required technical follow-up. Leave empty until the full topic is fetched and reviewed.

## Consistency rules

- Every discussion topic ID must also appear in the matching rank entry's `topic_ids`.
- Do not assign one topic to multiple ranks.
- Keep complementary posts from one team as separate discussion entries.
- Do not include a candidate when authorship or team association is materially ambiguous.
- Sort ranks numerically and discussions by rank, then topic ID.
