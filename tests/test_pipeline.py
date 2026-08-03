import json
import tempfile
import unittest
from pathlib import Path

from scripts.quality_gate import evaluate_quality_gate, promote_candidate
from scripts.run_pipeline import (
    enrich_metrics,
    evaluation_version,
    fingerprint_dataset,
    prepare_output_directory,
    validate_dataset,
    validate_model_names,
)


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
    def test_validate_model_names_accepts_any_non_empty_model_list(self) -> None:
        self.assertEqual(
            validate_model_names(["EfficientNetV2-S", "CustomNet"]),
            ["EfficientNetV2-S", "CustomNet"],
        )

    def test_validate_model_names_rejects_empty_or_duplicate_values(self) -> None:
        for value in ([], ["CustomNet", "CustomNet"]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_model_names(value)

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

    def test_prepare_output_directory_removes_stale_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "outputs" / "latest"
            output_dir.mkdir(parents=True)
            (output_dir / "stale.pth").write_bytes(b"old")

            prepare_output_directory(root, output_dir)

            self.assertTrue(output_dir.is_dir())
            self.assertFalse((output_dir / "stale.pth").exists())

    def test_prepare_output_directory_rejects_broad_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            with self.assertRaisesRegex(ValueError, "must be a subdirectory"):
                prepare_output_directory(root, root / "outputs")

    def test_enrich_metrics_rejects_unknown_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_dataset(root)
            output_dir = root / "outputs" / "latest"
            output_dir.mkdir(parents=True)
            metrics_path = root / "metrics" / "candidate.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": [
                            {
                                "model_name": "EfficientNetV2-S",
                                "best_val_accuracy": 0.96,
                                "test_accuracy": 0.95,
                                "test_macro_f1": 0.95,
                                "checkpoint": "model.pth",
                            }
                        ],
                        "candidate": {"model_name": "UnknownModel"},
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "train": {
                    "models": ["EfficientNetV2-S"],
                    "seed": 42,
                    "batch_size": 32,
                    "epochs": 1,
                    "num_workers": 0,
                    "val_ratio": 0.15,
                }
            }

            with self.assertRaisesRegex(ValueError, "UnknownModel"):
                enrich_metrics(
                    metrics_path,
                    root,
                    output_dir,
                    config,
                    {"dataset_dir": str(dataset)},
                )

    def test_enrich_metrics_accepts_three_configured_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_dataset(root)
            output_dir = root / "outputs" / "latest"
            output_dir.mkdir(parents=True)
            metrics_path = root / "metrics" / "candidate.json"
            metrics_path.parent.mkdir(parents=True)
            model_names = ["ModelA", "ModelB", "ModelC"]
            models = [
                {
                    "model_name": name,
                    "best_val_accuracy": 0.90 + index * 0.01,
                    "test_accuracy": 0.89 + index * 0.01,
                    "test_macro_f1": 0.88 + index * 0.01,
                    "checkpoint": f"{name}.pth",
                }
                for index, name in enumerate(model_names)
            ]
            metrics_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": models,
                        "candidate": models[-1],
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "train": {
                    "models": model_names,
                    "seed": 42,
                    "batch_size": 32,
                    "epochs": 1,
                    "num_workers": 0,
                    "val_ratio": 0.15,
                }
            }

            enriched = enrich_metrics(
                metrics_path,
                root,
                output_dir,
                config,
                {"dataset_dir": str(dataset)},
            )

        self.assertEqual(enriched["candidate"]["model_name"], "ModelC")
        self.assertEqual(enriched["train_params"]["models"], model_names)
        self.assertEqual(len(enriched["models"]), 3)

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
