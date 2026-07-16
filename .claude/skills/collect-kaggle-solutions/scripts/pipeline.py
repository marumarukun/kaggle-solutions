from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from jinja2 import Template
from markdown_it import MarkdownIt
from pypdf import PdfReader
from weasyprint import HTML


SKILL_DIR = Path(__file__).resolve().parents[1]
FONT_PATH = SKILL_DIR / "assets" / "NotoSansJP-Variable.ttf"


PAGE_TEMPLATE = Template(
    """<!doctype html>
<html lang="{{ lang }}">
<head>
<meta charset="utf-8">
<title>{{ document_title }}</title>
<style>
@font-face {
  font-family: "Noto Sans JP";
  src: url("{{ font_uri }}") format("truetype");
  font-weight: 100 900;
}
@page {
  size: A4;
  margin: 18mm 17mm 19mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    color: #667085;
    font-size: 8.5pt;
  }
}
html { font-family: "Noto Sans JP", sans-serif; color: #182230; }
body { font-size: 9.5pt; line-height: 1.65; overflow-wrap: anywhere; }
.cover { page-break-after: always; padding-top: 28mm; }
.eyebrow { color: #175cd3; font-size: 10pt; font-weight: 700; letter-spacing: .04em; }
h1 { font-size: 25pt; line-height: 1.3; margin: 8mm 0; color: #101828; }
h2 { font-size: 16pt; line-height: 1.4; margin: 8mm 0 3mm; border-bottom: 1px solid #d0d5dd; padding-bottom: 1.5mm; }
h3 { font-size: 12.5pt; margin: 6mm 0 2mm; }
h4 { font-size: 10.5pt; margin: 4mm 0 1mm; }
.meta { border-left: 4px solid #2e90fa; padding: 3mm 5mm; background: #eff8ff; }
.meta div { margin: 1.3mm 0; }
.label { display: inline-block; width: 25mm; color: #475467; font-weight: 700; }
.source { margin-top: 10mm; font-size: 8.5pt; color: #475467; }
a { color: #175cd3; text-decoration: none; }
p { margin: 2.5mm 0; }
ul, ol { padding-left: 6mm; }
li { margin: 1mm 0; }
img { display: block; max-width: 100%; max-height: 225mm; object-fit: contain; margin: 4mm auto; }
table { border-collapse: collapse; width: 100%; margin: 4mm 0; font-size: 8.2pt; }
th, td { border: 1px solid #98a2b3; padding: 1.6mm 2mm; vertical-align: top; }
th { background: #f2f4f7; }
pre { white-space: pre-wrap; background: #101828; color: #f2f4f7; padding: 3mm; border-radius: 2mm; font-size: 7.7pt; line-height: 1.45; }
code { font-family: "DejaVu Sans Mono", monospace; background: #f2f4f7; padding: 0 .6mm; }
pre code { background: transparent; padding: 0; }
blockquote { border-left: 3px solid #98a2b3; margin-left: 0; padding-left: 4mm; color: #475467; }
.content > :first-child { margin-top: 0; }
.translation-note { font-size: 8.5pt; color: #475467; margin-bottom: 6mm; }
.qa-message { border-left: 3px solid #84adff; padding-left: 4mm; margin: 5mm 0; }
.qa-meta { color: #475467; font-size: 8.5pt; font-weight: 700; }
</style>
</head>
<body>
<section class="cover">
  <div class="eyebrow">{{ competition }}</div>
  <h1>{{ document_title }}</h1>
  <div class="meta">
    <div><span class="label">Rank</span>{{ rank }}</div>
    <div><span class="label">Team</span>{{ team }}</div>
    <div><span class="label">Author</span>{{ author }}</div>
    <div><span class="label">Topic ID</span>{{ topic_id }}</div>
    <div><span class="label">Version</span>{{ version_label }}</div>
  </div>
  <div class="source">Source: <a href="{{ source_url }}">{{ source_url }}</a><br>
  Retrieved with Kaggle CLI on {{ retrieval_date }}.</div>
</section>
<main>
  {% if translation_note %}<div class="translation-note">{{ translation_note }}</div>{% endif %}
  <div class="content">{{ content_html }}</div>
</main>
</body>
</html>"""
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_text(value: str) -> str:
    """Collapse whitespace so names wrapped across PDF lines still match."""
    return re.sub(r"\s+", "", value)


def normalize_competition(value: str) -> str:
    value = value.strip()
    match = re.fullmatch(
        r"https?://(?:www\.)?kaggle\.com/competitions/([^/?#]+)/?", value
    )
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        return value
    raise ValueError(
        "competition must be a Kaggle competition slug or a full /competitions/<slug> URL"
    )


def find_kaggle() -> str:
    executable = shutil.which("kaggle")
    if not executable:
        raise RuntimeError(
            "Kaggle CLI was not found. Run this script through the project's uv environment."
        )
    return executable


def run_kaggle_json(*args: str) -> tuple[object, str]:
    result = subprocess.run(
        [find_kaggle(), *args, "--format", "json"],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Kaggle CLI failed: {detail}")
    if result.stdout.strip() == "No topics found":
        return [], result.stderr
    decoder = json.JSONDecoder()
    for match in re.finditer(r"(?m)^[\[{]", result.stdout):
        try:
            value, end = decoder.raw_decode(result.stdout[match.start() :])
            auxiliary = (
                result.stdout[: match.start()]
                + result.stdout[match.start() + end :]
                + result.stderr
            )
            return value, auxiliary
        except json.JSONDecodeError:
            continue
    raise RuntimeError(
        f"Kaggle CLI did not return JSON. stdout={result.stdout[:500]!r}"
    )


def next_page_token(stderr: str) -> str | None:
    match = re.search(r"Next Page Token\s*=\s*(\S+)", stderr or "")
    return match.group(1) if match else None


def competition_end_month(metadata: dict) -> str:
    deadline = str(metadata.get("deadline", "")).strip()
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})(?:T|$)", deadline)
    if not match:
        raise RuntimeError(
            "Competition metadata has no valid deadline; cannot determine the "
            "YYYYMM solution directory prefix"
        )
    try:
        end_date = datetime.strptime("-".join(match.groups()), "%Y-%m-%d")
    except ValueError as error:
        raise RuntimeError(
            f"Competition metadata has an invalid deadline {deadline!r}"
        ) from error
    return end_date.strftime("%Y%m")


def competition_directory_name(slug: str, metadata: dict) -> str:
    return f"{competition_end_month(metadata)}-{slug}"


def find_existing_base(project_root: Path, slug: str) -> Path | None:
    solutions_root = project_root / "solutions"
    if not solutions_root.is_dir():
        return None
    pattern = re.compile(rf"\d{{4}}(?:0[1-9]|1[0-2])-{re.escape(slug)}")
    candidates = sorted(
        path
        for path in solutions_root.iterdir()
        if path.is_dir() and pattern.fullmatch(path.name)
    )
    if len(candidates) > 1:
        relative_candidates = ", ".join(
            str(path.relative_to(project_root)) for path in candidates
        )
        raise RuntimeError(
            f"Multiple solution directories match competition {slug!r}: "
            f"{relative_candidates}"
        )
    return candidates[0] if candidates else None


def paths_from_base(base: Path) -> dict[str, Path]:
    work = base / ".work"
    return {
        "base": base,
        "work": work,
        "competition": work / "competition.json",
        "leaderboard": work / "leaderboard.json",
        "leaderboard_raw": work / "leaderboard-raw.json",
        "leaderboard_anomalies": work / "leaderboard-anomalies.json",
        "topics": work / "topics.json",
        "manifest": work / "manifest.json",
        "raw": work / "raw",
        "translations": work / "translations",
        "assets": work / "assets",
        "html": work / "html",
        "verification": work / "verification",
        "article_evidence": work / "article-evidence.md",
        "article": base / "article.md",
        "pdf": base / "pdf",
    }


def paths(project_root: Path, slug: str) -> dict[str, Path]:
    base = find_existing_base(project_root, slug)
    if base is None:
        raise FileNotFoundError(
            f"No collected state found for {slug!r} under "
            f"{project_root / 'solutions'}. Run collect first."
        )
    return paths_from_base(base)


def paths_for_metadata(project_root: Path, slug: str, metadata: dict) -> dict[str, Path]:
    base = project_root / "solutions" / competition_directory_name(slug, metadata)
    existing_base = find_existing_base(project_root, slug)
    if existing_base is not None and existing_base != base:
        raise RuntimeError(
            f"Existing directory {existing_base.relative_to(project_root)} conflicts "
            f"with competition deadline {metadata.get('deadline')!r}, which resolves "
            f"to {base.relative_to(project_root)}"
        )
    return paths_from_base(base)


SUBMISSION_FILE_EXTENSIONS = (
    "arrow",
    "csv",
    "feather",
    "gz",
    "json",
    "jsonl",
    "parquet",
    "tsv",
    "xlsx",
    "zip",
)


def leaderboard_anomaly_reasons(row: dict) -> list[str]:
    """Return reasons why a leaderboard row is not a plausible ranked team."""
    reasons: list[str] = []
    raw_team_name = row.get("teamName")
    team_name = str(raw_team_name).strip() if raw_team_name is not None else ""
    if not team_name:
        reasons.append("teamName is empty")
    extension_pattern = "|".join(re.escape(value) for value in SUBMISSION_FILE_EXTENSIONS)
    if re.search(rf"\.(?:{extension_pattern})$", team_name, re.IGNORECASE):
        reasons.append("teamName looks like a submission filename")
    return reasons


def leaderboard_failures(rows: object, max_rank: int) -> list[str]:
    failures: list[str] = []
    if not isinstance(rows, list):
        return ["leaderboard must be a JSON list"]
    if len(rows) != max_rank:
        failures.append(
            f"leaderboard has {len(rows)} rows; expected exactly {max_rank}"
        )
    expected_ranks = list(range(1, max_rank + 1))
    actual_ranks = [row.get("rank") for row in rows if isinstance(row, dict)]
    if actual_ranks != expected_ranks:
        failures.append(
            f"leaderboard ranks are {actual_ranks}; expected {expected_ranks}"
        )
    team_ids: list[object] = []
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            failures.append(f"leaderboard row {position} is not an object")
            continue
        reasons = leaderboard_anomaly_reasons(row)
        if reasons:
            failures.append(
                f"leaderboard rank {position} is suspicious: {', '.join(reasons)}"
            )
        team_id = row.get("teamId")
        if team_id is not None:
            if team_id in team_ids:
                failures.append(f"leaderboard teamId {team_id!r} appears more than once")
            team_ids.append(team_id)
    return failures


def manifest_leaderboard_failures(manifest: dict, leaderboard: list[dict]) -> list[str]:
    failures: list[str] = []
    leaderboard_by_rank = {row.get("rank"): row for row in leaderboard}
    for entry in manifest.get("ranks", []):
        number = entry.get("rank")
        source = leaderboard_by_rank.get(number)
        if source is None:
            failures.append(f"manifest rank {number!r} has no leaderboard row")
            continue
        expected_team = source.get("teamName", "")
        expected_score = source.get("score", "")
        if entry.get("team") != expected_team:
            failures.append(
                f"manifest rank {number} team {entry.get('team')!r} does not match "
                f"leaderboard team {expected_team!r}"
            )
        if entry.get("private_score") != expected_score:
            failures.append(
                f"manifest rank {number} score {entry.get('private_score')!r} does not "
                f"match leaderboard score {expected_score!r}"
            )
    return failures


def collect_leaderboard(
    slug: str, max_rank: int
) -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    raw_rows: list[dict] = []
    anomalies: list[dict] = []
    token: str | None = None
    seen_tokens: set[str] = set()
    while len(rows) < max_rank:
        page_size = min(200, max_rank - len(rows))
        args = [
            "competitions",
            "leaderboard",
            slug,
            "--show",
            "--page-size",
            str(page_size),
        ]
        if token:
            args.extend(["--page-token", token])
        page, stderr = run_kaggle_json(*args)
        if not isinstance(page, list) or not page:
            break
        for row in page:
            if not isinstance(row, dict):
                anomalies.append(
                    {
                        "source_position": len(raw_rows) + 1,
                        "row": row,
                        "reasons": ["row is not a JSON object"],
                    }
                )
                continue
            source_row = {**row, "source_position": len(raw_rows) + 1}
            raw_rows.append(source_row)
            reasons = leaderboard_anomaly_reasons(row)
            if reasons:
                anomalies.append({**source_row, "reasons": reasons})
            else:
                rows.append(source_row)
                if len(rows) == max_rank:
                    break
        token = next_page_token(stderr)
        if not token:
            break
        if token in seen_tokens:
            raise RuntimeError("Leaderboard pagination repeated a page token")
        seen_tokens.add(token)
    if len(rows) < max_rank:
        raise RuntimeError(
            f"Official leaderboard returned only {len(rows)} plausible team rows; "
            f"requested rank {max_rank}. Review leaderboard-anomalies.json."
        )
    ranked_rows = [
        {**row, "rank": rank}
        for rank, row in enumerate(rows[:max_rank], start=1)
    ]
    failures = leaderboard_failures(ranked_rows, max_rank)
    if failures:
        raise RuntimeError("Leaderboard validation failed:\n" + "\n".join(failures))
    return ranked_rows, raw_rows, anomalies


def collect_topics(slug: str) -> list[dict]:
    by_id: dict[int, dict] = {}
    for page_number in range(1, 501):
        page, _ = run_kaggle_json(
            "competitions",
            "topics",
            "list",
            slug,
            "--page",
            str(page_number),
            "--sort-by",
            "recent",
        )
        if not isinstance(page, list) or not page:
            break
        before = len(by_id)
        for topic in page:
            if topic.get("id") is not None:
                by_id[int(topic["id"])] = topic
        if len(by_id) == before:
            break
    else:
        raise RuntimeError("Discussion pagination exceeded 500 pages")
    return sorted(by_id.values(), key=lambda item: str(item.get("postDate", "")))


def collect(args: argparse.Namespace) -> None:
    slug = normalize_competition(args.competition)
    if args.max_rank < 1:
        raise ValueError("max-rank must be a positive integer")
    project_root = Path(args.project_root).resolve()

    matches, _ = run_kaggle_json(
        "competitions", "list", "--search", slug, "--page-size", "200"
    )
    exact = [
        item
        for item in matches
        if str(item.get("ref", "")).rstrip("/").endswith("/" + slug)
    ]
    if len(exact) != 1:
        raise RuntimeError(
            f"Could not uniquely identify competition {slug!r} through Kaggle CLI"
        )
    metadata = exact[0]
    target = paths_for_metadata(project_root, slug, metadata)
    target["work"].mkdir(parents=True, exist_ok=True)
    pages_data, _ = run_kaggle_json(
        "competitions", "pages", slug, "--content"
    )
    leaderboard, raw_leaderboard, leaderboard_anomalies = collect_leaderboard(
        slug, args.max_rank
    )
    topics = collect_topics(slug)
    candidate_pattern = re.compile(
        r"\b(solution|write[ -]?up|place|gold|approach)\b", re.IGNORECASE
    )
    candidate_ids = [
        int(item["id"])
        for item in topics
        if candidate_pattern.search(str(item.get("title", "")))
    ]
    write_json(
        target["competition"],
        {
            "slug": slug,
            "url": f"https://www.kaggle.com/competitions/{slug}",
            "max_rank": args.max_rank,
            "retrieved_at": now_iso(),
            "leaderboard_anomaly_count": len(leaderboard_anomalies),
            "metadata": metadata,
            "pages": pages_data,
        },
    )
    write_json(target["leaderboard_raw"], raw_leaderboard)
    write_json(target["leaderboard_anomalies"], leaderboard_anomalies)
    write_json(target["leaderboard"], leaderboard)
    write_json(
        target["topics"],
        {
            "retrieved_at": now_iso(),
            "topics": topics,
            "candidate_topic_ids": candidate_ids,
        },
    )
    print(f"Collected {len(leaderboard)} leaderboard rows and {len(topics)} topics")
    if leaderboard_anomalies:
        print(
            f"WARNING: excluded {len(leaderboard_anomalies)} suspicious leaderboard "
            f"row(s); review {target['leaderboard_anomalies'].relative_to(project_root)}"
        )
    print(f"Work directory: {target['work'].relative_to(project_root)}")


def init_manifest(args: argparse.Namespace) -> None:
    slug = normalize_competition(args.competition)
    project_root = Path(args.project_root).resolve()
    target = paths(project_root, slug)
    if target["manifest"].exists() and not args.overwrite:
        raise FileExistsError(
            f"Manifest already exists: {target['manifest']}. Use --overwrite explicitly."
        )
    competition = read_json(target["competition"])
    leaderboard = read_json(target["leaderboard"])
    failures = leaderboard_failures(leaderboard, competition["max_rank"])
    if failures:
        raise RuntimeError("Leaderboard validation failed:\n" + "\n".join(failures))
    manifest = {
        "competition": slug,
        "competition_url": competition["url"],
        "max_rank": competition["max_rank"],
        "created_at": now_iso(),
        "ranks": [
            {
                "rank": row["rank"],
                "team": row.get("teamName", ""),
                "private_score": row.get("score", ""),
                "status": "not_reviewed",
                "topic_ids": [],
            }
            for row in leaderboard
        ],
        "discussions": [],
    }
    write_json(target["manifest"], manifest)
    print(f"Created {target['manifest'].relative_to(project_root)}")


REQUIRED_DISCUSSION_FIELDS = (
    "rank",
    "team",
    "topic_id",
    "slug",
    "match_evidence",
    "selected_comment_ids",
)


def manifest_failures(manifest: dict) -> list[str]:
    failures: list[str] = []
    max_rank = manifest.get("max_rank")
    if not isinstance(max_rank, int) or max_rank < 1:
        failures.append(f"max_rank must be a positive integer, got {max_rank!r}")
        max_rank = 0

    rank_numbers: list[int] = []
    rank_by_number: dict[int, dict] = {}
    for entry in manifest.get("ranks", []):
        number = entry.get("rank")
        if not isinstance(number, int):
            failures.append(f"rank entry has a non-integer rank: {number!r}")
            continue
        rank_numbers.append(number)
        if number in rank_by_number:
            failures.append(f"duplicate rank entry: {number}")
        else:
            rank_by_number[number] = entry
    if rank_numbers != sorted(rank_numbers):
        failures.append("rank entries are not sorted numerically")
    missing_ranks = [n for n in range(1, max_rank + 1) if n not in rank_by_number]
    if missing_ranks:
        failures.append(f"missing rank entries: {missing_ranks}")
    outside = sorted(n for n in rank_by_number if not 1 <= n <= max_rank)
    if outside:
        failures.append(f"rank entries outside 1..{max_rank}: {outside}")

    rank_topic_owner: dict[int, int] = {}
    for number, entry in sorted(rank_by_number.items()):
        status = entry.get("status")
        topic_ids = entry.get("topic_ids")
        if not isinstance(topic_ids, list) or not all(
            isinstance(topic_id, int) for topic_id in topic_ids
        ):
            failures.append(f"rank {number} topic_ids must be a list of integers")
            topic_ids = []
        if status not in {"found", "not_found"}:
            failures.append(
                f"rank {number} has unresolved status {status!r}; "
                "finish the review with found or not_found"
            )
        if status == "found" and not topic_ids:
            failures.append(f"rank {number} is found but lists no topic_ids")
        if status == "not_found" and topic_ids:
            failures.append(f"rank {number} is not_found but lists topic_ids {topic_ids}")
        if len(set(topic_ids)) != len(topic_ids):
            failures.append(f"rank {number} lists duplicate topic_ids")
        for topic_id in topic_ids:
            if topic_id in rank_topic_owner and rank_topic_owner[topic_id] != number:
                failures.append(
                    f"topic {topic_id} is assigned to ranks "
                    f"{rank_topic_owner[topic_id]} and {number}"
                )
            rank_topic_owner.setdefault(topic_id, number)

    discussion_keys: list[tuple[int, int]] = []
    discussion_topics_by_rank: dict[int, set[int]] = {}
    for index, discussion in enumerate(manifest.get("discussions", [])):
        label = f"discussion #{index + 1}"
        missing_fields = [
            field for field in REQUIRED_DISCUSSION_FIELDS if field not in discussion
        ]
        if missing_fields:
            failures.append(f"{label} is missing fields: {', '.join(missing_fields)}")
            continue
        number = discussion["rank"]
        topic_id = discussion["topic_id"]
        if not isinstance(number, int) or not isinstance(topic_id, int):
            failures.append(f"{label} rank and topic_id must be integers")
            continue
        label = f"discussion rank {number} topic {topic_id}"
        if topic_id in discussion_topics_by_rank.get(number, set()):
            failures.append(f"{label} appears more than once")
        discussion_keys.append((number, topic_id))
        discussion_topics_by_rank.setdefault(number, set()).add(topic_id)
        entry = rank_by_number.get(number)
        if entry is None:
            failures.append(f"{label} has no matching rank entry")
        else:
            if discussion["team"] != entry.get("team"):
                failures.append(
                    f"{label} team {discussion['team']!r} does not match "
                    f"leaderboard team {entry.get('team')!r}"
                )
            if topic_id not in (entry.get("topic_ids") or []):
                failures.append(f"{label} is missing from the rank entry's topic_ids")
        slug = discussion["slug"]
        if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
            failures.append(
                f"{label} slug must be lowercase ASCII letters, digits, and hyphens"
            )
        evidence = discussion["match_evidence"]
        if not isinstance(evidence, str) or not evidence.strip():
            failures.append(f"{label} match_evidence must explain the rank-to-post match")
        selected = discussion["selected_comment_ids"]
        if not isinstance(selected, list) or not all(
            isinstance(value, int) for value in selected
        ):
            failures.append(f"{label} selected_comment_ids must be a list of integers")
        elif len(set(selected)) != len(selected):
            failures.append(f"{label} selected_comment_ids contains duplicates")
    if discussion_keys != sorted(discussion_keys):
        failures.append("discussions are not sorted by rank, then topic ID")

    for number, entry in sorted(rank_by_number.items()):
        topic_ids = entry.get("topic_ids")
        if not isinstance(topic_ids, list):
            continue
        for topic_id in topic_ids:
            if isinstance(topic_id, int) and topic_id not in discussion_topics_by_rank.get(
                number, set()
            ):
                failures.append(
                    f"rank {number} lists topic {topic_id} without a matching discussion entry"
                )
    return failures


def check_manifest(args: argparse.Namespace) -> None:
    slug = normalize_competition(args.competition)
    project_root = Path(args.project_root).resolve()
    target = paths(project_root, slug)
    manifest = read_json(target["manifest"])
    failures = manifest_failures(manifest)
    leaderboard = read_json(target["leaderboard"])
    failures.extend(leaderboard_failures(leaderboard, manifest.get("max_rank", 0)))
    failures.extend(manifest_leaderboard_failures(manifest, leaderboard))
    if failures:
        raise RuntimeError("Manifest check failed:\n" + "\n".join(failures))
    found = sum(
        1 for entry in manifest.get("ranks", []) if entry.get("status") == "found"
    )
    print(
        f"Manifest OK: {len(manifest.get('ranks', []))} ranks ({found} found), "
        f"{len(manifest.get('discussions', []))} discussions"
    )


def fetch_topic(slug: str, topic_id: int) -> dict:
    messages, _ = run_kaggle_json(
        "competitions",
        "topic-messages",
        slug,
        str(topic_id),
        "--sort-by",
        "old",
        "--page-size",
        "1",
    )
    if not isinstance(messages, list) or not messages:
        raise RuntimeError(f"Topic {topic_id} returned no root message")

    comments_by_id: dict[int, dict] = {}
    token: str | None = None
    topic: dict | None = None
    while True:
        command = [
            "competitions",
            "topics",
            "show",
            slug,
            str(topic_id),
            "--page-size",
            "200",
        ]
        if token:
            command.extend(["--page-token", token])
        payload, stderr = run_kaggle_json(*command)
        if topic is None:
            topic = payload.get("topic", {})
        for comment in payload.get("comments", []):
            comments_by_id[int(comment["id"])] = comment
        token = next_page_token(stderr)
        if not token:
            break
    comments = sorted(
        comments_by_id.values(), key=lambda item: str(item.get("postDate", ""))
    )
    return {
        "topic": topic or {},
        "source_url": f"https://www.kaggle.com/competitions/{slug}/discussion/{topic_id}",
        "retrieved_at": now_iso(),
        "main": messages[0],
        "comments": comments,
    }


def fetch(args: argparse.Namespace) -> None:
    slug = normalize_competition(args.competition)
    project_root = Path(args.project_root).resolve()
    target = paths(project_root, slug)
    manifest = read_json(target["manifest"])
    failures = manifest_failures(manifest)
    if failures:
        raise RuntimeError(
            "Manifest check failed; fix .work/manifest.json first:\n"
            + "\n".join(failures)
        )
    discussions = manifest.get("discussions", [])
    if not discussions:
        raise RuntimeError(
            "Manifest contains no discussions. Review leaderboard/topics and populate it first."
        )
    target["raw"].mkdir(parents=True, exist_ok=True)
    for discussion in discussions:
        topic_id = int(discussion["topic_id"])
        output = target["raw"] / f"{topic_id}.json"
        if output.exists() and not args.refresh:
            print(f"Reusing {output.relative_to(project_root)}")
            continue
        payload = fetch_topic(slug, topic_id)
        write_json(output, payload)
        print(f"Fetched topic {topic_id}")


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized[:60] or "solution"


def discussion_filename(discussion: dict, raw: dict) -> str:
    title = discussion.get("slug") or raw.get("topic", {}).get("title", "solution")
    return (
        f"rank-{int(discussion['rank']):02d}-{slugify(str(title))}-"
        f"{int(discussion['topic_id'])}"
    )


def clean_html(fragment: str) -> str:
    soup = BeautifulSoup(fragment or "", "html.parser")
    for tag in soup.find_all(["script", "style", "iframe", "form"]):
        tag.decompose()
    for tag in soup.find_all(True):
        for attribute in list(tag.attrs):
            if attribute.lower().startswith("on"):
                del tag.attrs[attribute]
    return str(soup)


def image_extension(url: str, content_type: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
        return suffix
    guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
    return guessed or ".img"


def localize_images(fragment: str, topic_id: int, asset_root: Path) -> str:
    soup = BeautifulSoup(fragment or "", "html.parser")
    topic_asset_dir = asset_root / str(topic_id)
    topic_asset_dir.mkdir(parents=True, exist_ok=True)
    for image in soup.find_all("img"):
        source = image.get("src")
        if not source or not source.startswith(("http://", "https://")):
            continue
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        existing = list(topic_asset_dir.glob(f"{digest}.*"))
        if existing:
            image["src"] = existing[0].resolve().as_uri()
            continue
        try:
            response = requests.get(source, timeout=45)
            response.raise_for_status()
            suffix = image_extension(source, response.headers.get("content-type", ""))
            destination = topic_asset_dir / f"{digest}{suffix}"
            destination.write_bytes(response.content)
            image["src"] = destination.resolve().as_uri()
        except requests.RequestException as error:
            print(f"Warning: could not download image {source}: {error}", file=sys.stderr)
    return str(soup)


def english_content(discussion: dict, raw: dict) -> str:
    content = clean_html(raw.get("main", {}).get("content", ""))
    selected = {int(value) for value in discussion.get("selected_comment_ids", [])}
    if not selected:
        return content
    comments = [
        comment for comment in raw.get("comments", []) if int(comment["id"]) in selected
    ]
    missing = selected - {int(comment["id"]) for comment in comments}
    if missing:
        raise RuntimeError(
            f"Selected comment IDs are absent from topic {discussion['topic_id']}: {sorted(missing)}"
        )
    blocks = [content, "<h2>Supplemental Q&amp;A</h2>"]
    for comment in comments:
        author = BeautifulSoup(str(comment.get("authorName", "")), "html.parser").get_text()
        date = str(comment.get("postDate", ""))
        blocks.append(
            '<section class="qa-message">'
            f'<div class="qa-meta">{author} · {date}</div>'
            f'{clean_html(comment.get("content", ""))}</section>'
        )
    return "\n".join(blocks)


def render_one(
    project_root: Path,
    target: dict[str, Path],
    manifest: dict,
    discussion: dict,
    language: str,
    overwrite: bool,
) -> Path:
    topic_id = int(discussion["topic_id"])
    raw = read_json(target["raw"] / f"{topic_id}.json")
    filename = discussion_filename(discussion, raw)
    pdf_path = target["pdf"] / language / f"{filename}-{language}.pdf"
    if pdf_path.exists() and not overwrite:
        print(f"Reusing {pdf_path.relative_to(project_root)}")
        return pdf_path

    title = str(raw.get("topic", {}).get("title") or discussion.get("title") or "Solution")
    if language == "en":
        document_title = title
        content_html = english_content(discussion, raw)
        version_label = "Original"
        translation_note = ""
    else:
        translation_path = target["translations"] / f"{filename}-ja.md"
        if not translation_path.exists():
            raise FileNotFoundError(
                f"Missing Japanese translation: {translation_path.relative_to(project_root)}"
            )
        markdown = translation_path.read_text(encoding="utf-8")
        content_html = MarkdownIt("commonmark", {"html": True}).enable("table").render(
            markdown
        )
        first_heading = re.search(r"^#\s+(.+)$", markdown, flags=re.MULTILINE)
        document_title = first_heading.group(1) if first_heading else f"{title}（日本語訳）"
        if first_heading:
            content_html = re.sub(r"<h1>.*?</h1>", "", content_html, count=1, flags=re.S)
        version_label = "Japanese Translation"
        translation_note = (
            "Kaggle Discussionの主投稿と、技術的補足を含む選定済みQ&Aを日本語へ翻訳した版です。"
            "code、URL、model名、識別子、数値は原文を保持しています。"
        )

    content_html = localize_images(content_html, topic_id, target["assets"])
    html = PAGE_TEMPLATE.render(
        lang=language,
        competition=manifest["competition"],
        document_title=document_title,
        rank=discussion["rank"],
        team=discussion["team"],
        author=raw.get("topic", {}).get("authorName", discussion.get("author", "")),
        topic_id=topic_id,
        version_label=version_label,
        source_url=raw["source_url"],
        retrieval_date=str(raw.get("retrieved_at", ""))[:10],
        translation_note=translation_note,
        content_html=content_html,
        font_uri=FONT_PATH.resolve().as_uri(),
    )
    html_path = target["html"] / language / f"{filename}-{language}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    HTML(string=html, base_url=str(project_root)).write_pdf(pdf_path)
    print(f"Rendered {pdf_path.relative_to(project_root)}")
    return pdf_path


def render(args: argparse.Namespace) -> None:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Missing font asset: {FONT_PATH}")
    slug = normalize_competition(args.competition)
    project_root = Path(args.project_root).resolve()
    target = paths(project_root, slug)
    manifest = read_json(target["manifest"])
    languages = ["en", "ja"] if args.language == "all" else [args.language]
    for discussion in manifest.get("discussions", []):
        for language in languages:
            render_one(
                project_root, target, manifest, discussion, language, args.overwrite
            )


ARTICLE_REQUIRED_HEADINGS = (
    "## はじめに",
    "## コンペ概要",
    "## 上位解法の全体像",
    "## 上位解法から見えた、特に重要な発見",
    "## うまくいかなかったアプローチ",
    "## まとめ",
    "## 参照した上位Solution",
)


def article_evidence_failures(evidence_path: Path, manifest: dict) -> list[str]:
    if not evidence_path.exists():
        return [f"missing article evidence worksheet: {evidence_path}"]

    evidence = evidence_path.read_text(encoding="utf-8")
    failures: list[str] = []
    if "## Cross-team matrix" not in evidence:
        failures.append("article evidence worksheet has no cross-team matrix")
    for rank_entry in manifest.get("ranks", []):
        if rank_entry.get("status") != "found":
            continue
        rank = int(rank_entry["rank"])
        team = str(rank_entry["team"])
        if f"## Rank {rank} — {team}" not in evidence:
            failures.append(
                f"article evidence worksheet has no block for rank {rank} team {team!r}"
            )
    return failures


def article_failures(article_path: Path, manifest: dict) -> list[str]:
    if not article_path.exists():
        return [f"missing article: {article_path}"]

    article = article_path.read_text(encoding="utf-8")
    failures: list[str] = []
    if not re.search(r"(?m)^# .+上位解法まとめ\s+—\s+.+$", article):
        failures.append("article title must state the competition and a central thesis")
    for heading in ARTICLE_REQUIRED_HEADINGS:
        if heading not in article:
            failures.append(f"article missing required heading: {heading}")
    if article.count("```mermaid") < 2:
        failures.append("article must contain at least two Mermaid diagrams")
    if article.count("```") % 2:
        failures.append("article has an unbalanced fenced code block")
    if article.count("> **") < 2:
        failures.append(
            "article must emphasize the central thesis in the introduction and conclusion"
        )

    discussions_by_rank: dict[int, list[int]] = {}
    for discussion in manifest.get("discussions", []):
        rank = int(discussion["rank"])
        topic_id = int(discussion["topic_id"])
        discussions_by_rank.setdefault(rank, []).append(topic_id)
        if f"/discussion/{topic_id}" not in article:
            failures.append(f"article does not link discovered topic {topic_id}")

    for rank, topic_ids in discussions_by_rank.items():
        citation_count = sum(
            article.count(f"/discussion/{topic_id}") for topic_id in topic_ids
        )
        if citation_count <= len(topic_ids):
            failures.append(
                f"rank {rank} has no inline body citation in addition to its reference link"
            )

    for rank_entry in manifest.get("ranks", []):
        if rank_entry.get("status") != "not_found":
            continue
        rank = int(rank_entry["rank"])
        if not re.search(rf"(?<!\d){rank}\s*位", article):
            failures.append(f"article does not disclose unresolved rank {rank}")
    return failures


def verify(args: argparse.Namespace) -> None:
    slug = normalize_competition(args.competition)
    project_root = Path(args.project_root).resolve()
    target = paths(project_root, slug)
    manifest = read_json(target["manifest"])
    failures: list[str] = manifest_failures(manifest)
    leaderboard = read_json(target["leaderboard"])
    failures.extend(leaderboard_failures(leaderboard, manifest.get("max_rank", 0)))
    failures.extend(manifest_leaderboard_failures(manifest, leaderboard))
    failures.extend(article_evidence_failures(target["article_evidence"], manifest))
    rows: list[dict] = []
    article_path = target["article"]
    failures.extend(article_failures(article_path, manifest))
    for discussion in manifest.get("discussions", []):
        topic_id = int(discussion["topic_id"])
        raw_path = target["raw"] / f"{topic_id}.json"
        if not raw_path.exists():
            failures.append(f"missing raw topic: {raw_path.relative_to(project_root)}")
            continue
        raw = read_json(raw_path)
        filename = discussion_filename(discussion, raw)
        selected_ids = {
            int(value) for value in discussion.get("selected_comment_ids", [])
        }
        selected_comments = [
            comment
            for comment in raw.get("comments", [])
            if int(comment["id"]) in selected_ids
        ]
        if selected_ids:
            translation_path = target["translations"] / f"{filename}-ja.md"
            if not translation_path.exists():
                failures.append(
                    f"missing translation: {translation_path.relative_to(project_root)}"
                )
            else:
                translation = translation_path.read_text(encoding="utf-8")
                compact_translation = compact_text(translation)
                if "補足Q&A" not in translation:
                    failures.append(
                        f"supplemental Q&A missing from translation: "
                        f"{translation_path.relative_to(project_root)}"
                    )
                for comment in selected_comments:
                    author = str(comment.get("authorName", "")).strip()
                    if author and compact_text(author) not in compact_translation:
                        failures.append(
                            f"Q&A speaker {author!r} missing from translation: "
                            f"{translation_path.relative_to(project_root)}"
                        )
        for language in ("en", "ja"):
            pdf_path = target["pdf"] / language / f"{filename}-{language}.pdf"
            if not pdf_path.exists():
                failures.append(f"missing PDF: {pdf_path.relative_to(project_root)}")
                continue
            reader = PdfReader(pdf_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            text = text.replace("\x00", "")
            compact_pdf = compact_text(text)
            if not text.strip():
                failures.append(f"no extractable text: {pdf_path.relative_to(project_root)}")
            if compact_text(str(discussion["team"])) not in compact_pdf:
                failures.append(f"team missing from PDF: {pdf_path.relative_to(project_root)}")
            if language == "ja" and not re.search(r"[ぁ-んァ-ヶ一-龯]", text):
                failures.append(f"Japanese text missing: {pdf_path.relative_to(project_root)}")
            for comment in selected_comments:
                author = str(comment.get("authorName", "")).strip()
                if author and compact_text(author) not in compact_pdf:
                    failures.append(
                        f"Q&A speaker {author!r} missing from PDF: "
                        f"{pdf_path.relative_to(project_root)}"
                    )
            rows.append(
                {
                    "file": str(pdf_path.relative_to(project_root)),
                    "pages": len(reader.pages),
                    "bytes": pdf_path.stat().st_size,
                    "text_characters": len(text),
                }
            )
    report = {
        "competition": slug,
        "verified_at": now_iso(),
        "expected_discussions": len(manifest.get("discussions", [])),
        "files": rows,
        "failures": failures,
    }
    report_path = target["verification"] / "report.json"
    write_json(report_path, report)
    if failures:
        raise RuntimeError("Verification failed:\n" + "\n".join(failures))
    print(f"Verified article and {len(rows)} PDFs")
    print(f"Report: {report_path.relative_to(project_root)}")


def status(args: argparse.Namespace) -> None:
    slug = normalize_competition(args.competition)
    project_root = Path(args.project_root).resolve()
    base = find_existing_base(project_root, slug)
    if base is None:
        print(
            json.dumps(
                {
                    "competition": slug,
                    "output_directory": None,
                    "competition_collected": False,
                    "leaderboard_collected": False,
                    "topics_collected": False,
                    "manifest_exists": False,
                    "article_evidence_exists": False,
                    "article_exists": False,
                    "english_pdfs": 0,
                    "japanese_pdfs": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    target = paths_from_base(base)
    result = {
        "competition": slug,
        "output_directory": str(base.relative_to(project_root)),
        "competition_collected": target["competition"].exists(),
        "leaderboard_collected": target["leaderboard"].exists(),
        "topics_collected": target["topics"].exists(),
        "manifest_exists": target["manifest"].exists(),
        "article_evidence_exists": target["article_evidence"].exists(),
        "article_exists": target["article"].exists(),
        "english_pdfs": len(list((target["pdf"] / "en").glob("*.pdf"))),
        "japanese_pdfs": len(list((target["pdf"] / "ja").glob("*.pdf"))),
    }
    if target["manifest"].exists():
        manifest = read_json(target["manifest"])
        expected = []
        for discussion in manifest.get("discussions", []):
            topic_id = int(discussion["topic_id"])
            raw_path = target["raw"] / f"{topic_id}.json"
            item = {
                "rank": discussion["rank"],
                "team": discussion["team"],
                "topic_id": topic_id,
                "raw_exists": raw_path.exists(),
                "selected_comment_ids": discussion.get("selected_comment_ids", []),
            }
            if raw_path.exists():
                raw = read_json(raw_path)
                filename = discussion_filename(discussion, raw)
                item["translation_file"] = str(
                    (target["translations"] / f"{filename}-ja.md").relative_to(
                        project_root
                    )
                )
            expected.append(item)
        result["discussions"] = expected
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect and render ranked Kaggle solution discussions"
    )
    parser.add_argument("--project-root", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("competition")
    collect_parser.add_argument("--max-rank", type=int, required=True)
    collect_parser.set_defaults(handler=collect)

    manifest_parser = subparsers.add_parser("init-manifest")
    manifest_parser.add_argument("competition")
    manifest_parser.add_argument("--overwrite", action="store_true")
    manifest_parser.set_defaults(handler=init_manifest)

    check_parser = subparsers.add_parser("check-manifest")
    check_parser.add_argument("competition")
    check_parser.set_defaults(handler=check_manifest)

    fetch_parser = subparsers.add_parser("fetch")
    fetch_parser.add_argument("competition")
    fetch_parser.add_argument("--refresh", action="store_true")
    fetch_parser.set_defaults(handler=fetch)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("competition")
    render_parser.add_argument("--language", choices=["en", "ja", "all"], default="all")
    render_parser.add_argument("--overwrite", action="store_true")
    render_parser.set_defaults(handler=render)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("competition")
    verify_parser.set_defaults(handler=verify)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("competition")
    status_parser.set_defaults(handler=status)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
