import json
import tempfile
import unittest
from pathlib import Path

from scripts.quality_gate import evaluate_quality_gate, promote_candidate
from scripts.run_pipeline import evaluation_version, fingerprint_dataset, validate_dataset


CLASSES = ["Healthy", "Miner", "Phoma", "Rust"]


def make_dataset(root: Path) -> Path:
    dataset = root / "data"
    for split in ("Train", "test"):
        for class_name in CLASSES:
            class_dir = dataset / split / class_name
            class_dir.mkdir(parents=True)
            (class_dir / "sample.jpg").write_bytes(b"not-decoded-by-fast-validation")
    return dataset


def candidate_payload(checkpoint: Path, score: float = 0.96) -> dict:
    return {
        "candidate": {
            "model_name": "EfficientNetV2-S",
            "best_val_accuracy": score,
            "checkpoint": str(checkpoint),
            "model_uri": "runs:/candidate/model/model.pth",
            "mlflow_run_id": "candidate",
        },
        "evaluation_version": "evaluation-v1",
        "git_sha": "abc123",
        "data_version": "data-v1",
    }


def champion_payload(score: float = 0.95) -> dict:
    return {
        "model_name": "EfficientNetV2-S",
        "best_val_accuracy": score,
        "evaluation_version": "evaluation-v1",
        "mlflow_run_id": "champion",
    }


def gate_config() -> dict:
    return {
        "minimum_val_accuracy": 0.90,
        "minimum_improvement": 0.0,
        "require_same_evaluation_version": True,
    }


class PipelineTests(unittest.TestCase):
    def test_validate_dataset_accepts_expected_structure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = validate_dataset(make_dataset(Path(directory)), CLASSES)

        self.assertEqual(summary["total_images"], 8)
        self.assertEqual(summary["classes"], CLASSES)

    def test_validate_dataset_rejects_missing_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_dataset(Path(directory))
            for item in (dataset / "test" / "Rust").iterdir():
                item.unlink()
            (dataset / "test" / "Rust").rmdir()

            with self.assertRaisesRegex(ValueError, "test classes"):
                validate_dataset(dataset, CLASSES)

    def test_evaluation_version_is_deterministic(self) -> None:
        self.assertEqual(
            evaluation_version("data-v1", 42, 0.15),
            evaluation_version("data-v1", 42, 0.15),
        )
        self.assertNotEqual(
            evaluation_version("data-v1", 42, 0.15),
            evaluation_version("data-v2", 42, 0.15),
        )

    def test_dataset_fingerprint_changes_with_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = make_dataset(Path(directory))
            first = fingerprint_dataset(dataset)
            (dataset / "Train" / "Healthy" / "sample.jpg").write_bytes(b"changed")
            second = fingerprint_dataset(dataset)

        self.assertNotEqual(first, second)

    def test_quality_gate_approves_strict_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate.pth"
            checkpoint.write_bytes(b"weights")
            report = evaluate_quality_gate(
                candidate_payload(checkpoint), champion_payload(), gate_config(), Path("/")
            )

        self.assertTrue(report["approved"])
        self.assertTrue(all(report["checks"].values()))

    def test_quality_gate_rejects_equal_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate.pth"
            checkpoint.write_bytes(b"weights")
            report = evaluate_quality_gate(
                candidate_payload(checkpoint, score=0.95),
                champion_payload(score=0.95),
                gate_config(),
                Path("/"),
            )

        self.assertFalse(report["approved"])
        self.assertFalse(report["checks"]["strict_improvement"])

    def test_quality_gate_rejects_changed_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate.pth"
            checkpoint.write_bytes(b"weights")
            candidate = candidate_payload(checkpoint)
            candidate["evaluation_version"] = "evaluation-v2"
            report = evaluate_quality_gate(
                candidate, champion_payload(), gate_config(), Path("/")
            )

        self.assertFalse(report["approved"])
        self.assertFalse(report["checks"]["same_evaluation_version"])

    def test_promote_candidate_updates_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "candidate.pth"
            checkpoint.write_bytes(b"weights")
            candidate = candidate_payload(checkpoint)
            report = evaluate_quality_gate(
                candidate, champion_payload(), gate_config(), Path("/")
            )
            champion_path = root / "champion.json"
            promote_candidate(report, candidate, champion_path)
            promoted = json.loads(champion_path.read_text(encoding="utf-8"))

        self.assertEqual(promoted["mlflow_run_id"], "candidate")
        self.assertEqual(promoted["evaluation_version"], "evaluation-v1")


if __name__ == "__main__":
    unittest.main()
