from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("pipeline.py")
SPEC = importlib.util.spec_from_file_location("collect_kaggle_pipeline", MODULE_PATH)
assert SPEC and SPEC.loader
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class LeaderboardValidationTest(unittest.TestCase):
    def test_submission_filename_is_anomaly(self) -> None:
        row = {
            "teamId": 1,
            "teamName": "perfect_submission.parquet",
            "score": "1.00000",
        }
        self.assertEqual(
            pipeline.leaderboard_anomaly_reasons(row),
            ["teamName looks like a submission filename"],
        )

    def test_normal_team_is_not_anomaly(self) -> None:
        row = {"teamId": 2, "teamName": "MIC-DKFZ", "score": "0.83173"}
        self.assertEqual(pipeline.leaderboard_anomaly_reasons(row), [])

    def test_shifted_manifest_is_rejected(self) -> None:
        leaderboard = [
            {"rank": 1, "teamName": "winner", "score": "0.9"},
            {"rank": 2, "teamName": "runner-up", "score": "0.8"},
        ]
        manifest = {
            "ranks": [
                {"rank": 1, "team": "artifact.csv", "private_score": "1.0"},
                {"rank": 2, "team": "winner", "private_score": "0.9"},
            ]
        }
        failures = pipeline.manifest_leaderboard_failures(manifest, leaderboard)
        self.assertTrue(any("rank 1 team" in failure for failure in failures))
        self.assertTrue(any("rank 2 team" in failure for failure in failures))

    def test_collect_excludes_anomaly_fetches_replacement_and_reranks(self) -> None:
        artifact = {
            "teamId": 1,
            "teamName": "perfect_submission.parquet",
            "score": "1.00000",
        }
        winner = {"teamId": 2, "teamName": "winner", "score": "0.90000"}
        runner_up = {"teamId": 3, "teamName": "runner-up", "score": "0.80000"}
        responses = [
            ([artifact, winner], "Next Page Token = next"),
            ([runner_up], ""),
        ]
        with patch.object(pipeline, "run_kaggle_json", side_effect=responses):
            rows, raw_rows, anomalies = pipeline.collect_leaderboard("example", 2)

        self.assertEqual([row["teamName"] for row in rows], ["winner", "runner-up"])
        self.assertEqual([row["rank"] for row in rows], [1, 2])
        self.assertEqual(len(raw_rows), 3)
        self.assertEqual(anomalies[0]["teamName"], "perfect_submission.parquet")

    def test_contiguous_valid_leaderboard_passes(self) -> None:
        rows = [
            {"rank": 1, "teamId": 10, "teamName": "winner", "score": "0.9"},
            {"rank": 2, "teamId": 11, "teamName": "runner-up", "score": "0.8"},
        ]
        self.assertEqual(pipeline.leaderboard_failures(rows, 2), [])

    def test_pdf_identity_comparison_ignores_unrendered_emoji(self) -> None:
        expected = "BirdCLEF+ 2026 Team🤗🤗🤗"
        extracted = "Team BirdCLEF+ 2026 Team"
        self.assertIn(
            pipeline.compact_identity(expected),
            pipeline.compact_identity(extracted),
        )


class SolutionDirectoryTest(unittest.TestCase):
    def test_directory_name_uses_deadline_month(self) -> None:
        metadata = {"deadline": "2025-10-14T23:59:00"}
        self.assertEqual(
            pipeline.competition_directory_name("example-competition", metadata),
            "202510-example-competition",
        )

    def test_missing_deadline_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no valid deadline"):
            pipeline.competition_end_month({})

    def test_existing_directory_is_found_by_slug(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            expected = project_root / "solutions" / "202412-example-competition"
            expected.mkdir(parents=True)

            self.assertEqual(
                pipeline.find_existing_base(project_root, "example-competition"),
                expected,
            )
            self.assertEqual(
                pipeline.paths(project_root, "example-competition")["article"],
                expected / "article.md",
            )

    def test_multiple_matching_directories_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            (project_root / "solutions" / "202411-example").mkdir(parents=True)
            (project_root / "solutions" / "202412-example").mkdir(parents=True)

            with self.assertRaisesRegex(RuntimeError, "Multiple solution directories"):
                pipeline.find_existing_base(project_root, "example")

    def test_status_reports_empty_state_before_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = StringIO()
            args = Namespace(competition="example", project_root=directory)

            with redirect_stdout(output):
                pipeline.status(args)

            result = json.loads(output.getvalue())
            self.assertIsNone(result["output_directory"])
            self.assertFalse(result["competition_collected"])
            self.assertFalse(result["article_evidence_exists"])
            self.assertFalse(result["article_exists"])


class ArticleEvidenceValidationTest(unittest.TestCase):
    def test_complete_evidence_worksheet_passes(self) -> None:
        manifest = {
            "ranks": [
                {"rank": 1, "team": "winner", "status": "found"},
                {"rank": 2, "team": "runner-up", "status": "not_found"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "article-evidence.md"
            evidence.write_text(
                "## Rank 1 — winner\n\n## Cross-team matrix\n", encoding="utf-8"
            )
            self.assertEqual(
                pipeline.article_evidence_failures(evidence, manifest), []
            )

    def test_missing_found_team_block_is_rejected(self) -> None:
        manifest = {
            "ranks": [{"rank": 1, "team": "winner", "status": "found"}]
        }
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "article-evidence.md"
            evidence.write_text("## Cross-team matrix\n", encoding="utf-8")
            failures = pipeline.article_evidence_failures(evidence, manifest)
            self.assertTrue(any("no block for rank 1" in item for item in failures))


class ArticleValidationTest(unittest.TestCase):
    def valid_article(self) -> str:
        headings = "\n\n".join(pipeline.ARTICLE_REQUIRED_HEADINGS)
        return f"""# Example 上位解法まとめ — central thesis

{headings}

> **opening thesis**

```mermaid
flowchart LR
    A --> B
```

```mermaid
flowchart LR
    B --> C
```

[body source](https://www.kaggle.com/competitions/example/discussion/100)
[reference](https://www.kaggle.com/competitions/example/discussion/100)

2位は未取得です。

> **transferable thesis**
"""

    def test_complete_article_passes(self) -> None:
        manifest = {
            "ranks": [
                {"rank": 1, "status": "found"},
                {"rank": 2, "status": "not_found"},
            ],
            "discussions": [{"rank": 1, "topic_id": 100}],
        }
        with tempfile.TemporaryDirectory() as directory:
            article = Path(directory) / "article.md"
            article.write_text(self.valid_article(), encoding="utf-8")
            self.assertEqual(pipeline.article_failures(article, manifest), [])

    def test_missing_inline_citation_and_rank_disclosure_are_rejected(self) -> None:
        manifest = {
            "ranks": [
                {"rank": 1, "status": "found"},
                {"rank": 2, "status": "not_found"},
            ],
            "discussions": [{"rank": 1, "topic_id": 100}],
        }
        with tempfile.TemporaryDirectory() as directory:
            article = Path(directory) / "article.md"
            value = self.valid_article().replace(
                "[body source](https://www.kaggle.com/competitions/example/discussion/100)\n",
                "",
            ).replace("2位は未取得です。", "未取得の順位があります。")
            article.write_text(value, encoding="utf-8")
            failures = pipeline.article_failures(article, manifest)
            self.assertTrue(any("no inline body citation" in item for item in failures))
            self.assertTrue(any("unresolved rank 2" in item for item in failures))


if __name__ == "__main__":
    unittest.main()
