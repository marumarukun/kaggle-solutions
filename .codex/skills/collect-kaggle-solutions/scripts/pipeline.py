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


def paths(project_root: Path, slug: str) -> dict[str, Path]:
    base = project_root / "outputs" / slug
    work = base / ".work"
    return {
        "base": base,
        "work": work,
        "competition": work / "competition.json",
        "leaderboard": work / "leaderboard.json",
        "topics": work / "topics.json",
        "manifest": work / "manifest.json",
        "raw": work / "raw",
        "translations": work / "translations",
        "assets": work / "assets",
        "html": work / "html",
        "verification": work / "verification",
        "summary": base / "summary.md",
        "pdf": base / "pdf",
    }


def collect_leaderboard(slug: str, max_rank: int) -> list[dict]:
    rows: list[dict] = []
    token: str | None = None
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
        rows.extend(page)
        token = next_page_token(stderr)
        if not token:
            break
    if len(rows) < max_rank:
        raise RuntimeError(
            f"Official leaderboard returned only {len(rows)} rows; requested rank {max_rank}."
        )
    return [
        {"rank": rank, **row}
        for rank, row in enumerate(rows[:max_rank], start=1)
    ]


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
    target = paths(project_root, slug)
    target["work"].mkdir(parents=True, exist_ok=True)

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
    pages_data, _ = run_kaggle_json(
        "competitions", "pages", slug, "--content"
    )
    leaderboard = collect_leaderboard(slug, args.max_rank)
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
            "metadata": exact[0],
            "pages": pages_data,
        },
    )
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


def verify(args: argparse.Namespace) -> None:
    slug = normalize_competition(args.competition)
    project_root = Path(args.project_root).resolve()
    target = paths(project_root, slug)
    manifest = read_json(target["manifest"])
    failures: list[str] = []
    rows: list[dict] = []
    if not target["summary"].exists():
        failures.append(f"missing summary: {target['summary'].relative_to(project_root)}")
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
                if "補足Q&A" not in translation:
                    failures.append(
                        f"supplemental Q&A missing from translation: "
                        f"{translation_path.relative_to(project_root)}"
                    )
                for comment in selected_comments:
                    author = str(comment.get("authorName", "")).strip()
                    if author and author not in translation:
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
            if not text.strip():
                failures.append(f"no extractable text: {pdf_path.relative_to(project_root)}")
            if str(discussion["team"]) not in text:
                failures.append(f"team missing from PDF: {pdf_path.relative_to(project_root)}")
            if language == "ja" and not re.search(r"[ぁ-んァ-ヶ一-龯]", text):
                failures.append(f"Japanese text missing: {pdf_path.relative_to(project_root)}")
            for comment in selected_comments:
                author = str(comment.get("authorName", "")).strip()
                if author and author not in text:
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
    print(f"Verified summary and {len(rows)} PDFs")
    print(f"Report: {report_path.relative_to(project_root)}")


def status(args: argparse.Namespace) -> None:
    slug = normalize_competition(args.competition)
    project_root = Path(args.project_root).resolve()
    target = paths(project_root, slug)
    result = {
        "competition": slug,
        "competition_collected": target["competition"].exists(),
        "leaderboard_collected": target["leaderboard"].exists(),
        "topics_collected": target["topics"].exists(),
        "manifest_exists": target["manifest"].exists(),
        "summary_exists": target["summary"].exists(),
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
