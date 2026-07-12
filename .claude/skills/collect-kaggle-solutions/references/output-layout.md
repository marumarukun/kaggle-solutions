# Output layout

Write each run under `outputs/<competition-slug>/`.

```text
outputs/<competition-slug>/
├── summary.md
├── pdf/
│   ├── en/
│   │   └── rank-<rank>-<slug>-<topic-id>-en.pdf
│   └── ja/
│       └── rank-<rank>-<slug>-<topic-id>-ja.pdf
└── .work/
    ├── competition.json
    ├── leaderboard-raw.json
    ├── leaderboard-anomalies.json
    ├── leaderboard.json
    ├── topics.json
    ├── manifest.json
    ├── raw/
    ├── selected/
    ├── translations/
    ├── assets/
    ├── html/
    └── verification/
```

`leaderboard-raw.json` preserves every retrieved CLI row in retrieval order. `leaderboard-anomalies.json` records excluded rows and reasons. `leaderboard.json` is the validated, contiguous rank mapping used by the manifest and filenames.

## Git policy

Treat these as final, trackable artifacts:

- `summary.md`
- English PDFs under `pdf/en/`
- Japanese PDFs under `pdf/ja/`

Treat `.work/` as reproducible local state and exclude it from Git. The project root `.gitignore` must contain:

```gitignore
outputs/*/.work/
```

Do not place the only copy of a final Markdown or PDF inside `.work/`.

## Resume and overwrite policy

- Reuse valid files in `.work/` when resuming an interrupted run.
- Do not overwrite an existing `summary.md`, translation Markdown, or final PDF merely because the command was invoked again.
- Regenerate an artifact when it is absent, fails validation, or the user explicitly requests refresh or overwrite.
- Record source topic IDs, retrieval timestamps, selected comment IDs, and file checksums in `.work/manifest.json`.
- Keep downloaded source images in `.work/assets/`; embed or reference them when rendering the final PDFs.
