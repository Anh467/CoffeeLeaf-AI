from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
REQUIRED_CONFIG_SECTIONS = {"data", "train", "output", "mlflow", "quality_gate"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    missing = REQUIRED_CONFIG_SECTIONS.difference(config)
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(sorted(missing))}")
    return config


def resolve_from_root(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def validate_dataset(dataset_dir: Path, expected_classes: list[str]) -> dict[str, Any]:
    train_dir = dataset_dir / "Train"
    test_dir = dataset_dir / "test"
    errors: list[str] = []

    if not train_dir.is_dir():
        errors.append(f"Missing training directory: {train_dir}")
    if not test_dir.is_dir():
        errors.append(f"Missing test directory: {test_dir}")
    if errors:
        raise ValueError("; ".join(errors))

    expected = set(expected_classes)
    train_classes = {item.name for item in train_dir.iterdir() if item.is_dir()}
    test_classes = {item.name for item in test_dir.iterdir() if item.is_dir()}

    if train_classes != expected:
        errors.append(
            f"Train classes must be {sorted(expected)}, got {sorted(train_classes)}"
        )
    if test_classes != expected:
        errors.append(
            f"test classes must be {sorted(expected)}, got {sorted(test_classes)}"
        )
    if train_classes != test_classes:
        errors.append("Train and test class sets do not match")

    counts: dict[str, dict[str, int]] = {"Train": {}, "test": {}}
    for split_name, split_dir in (("Train", train_dir), ("test", test_dir)):
        for class_name in sorted(expected):
            class_dir = split_dir / class_name
            count = (
                sum(
                    1
                    for item in class_dir.rglob("*")
                    if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
                )
                if class_dir.is_dir()
                else 0
            )
            counts[split_name][class_name] = count
            if count == 0:
                errors.append(f"No supported images in {class_dir}")

    if errors:
        raise ValueError("; ".join(errors))

    return {
        "dataset_dir": str(dataset_dir),
        "classes": sorted(expected),
        "image_counts": counts,
        "total_images": sum(sum(split.values()) for split in counts.values()),
    }


def fingerprint_dataset(dataset_dir: Path) -> str:
    digest = hashlib.sha256()
    image_paths = sorted(
        item
        for item in dataset_dir.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in image_paths:
        digest.update(image_path.relative_to(dataset_dir).as_posix().encode("utf-8"))
        with image_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def read_data_version(repo_root: Path, dataset_dir: Path) -> str:
    pointer = repo_root / "data" / "raw.dvc"
    if not pointer.exists():
        return fingerprint_dataset(dataset_dir)

    with pointer.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    outputs = payload.get("outs") or []
    if not outputs:
        return "unknown"

    output = outputs[0]
    for key in ("md5", "etag", "checksum"):
        if output.get(key):
            return f"{key}:{output[key]}"
    return fingerprint_dataset(dataset_dir)


def get_git_sha(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def evaluation_version(data_version: str, seed: int, val_ratio: float) -> str:
    contract = f"{data_version}|seed={seed}|val_ratio={val_ratio:.12g}"
    return hashlib.sha256(contract.encode("utf-8")).hexdigest()


def execute_notebook(
    notebook_path: Path,
    output_dir: Path,
    metrics_path: Path,
    dataset_dir: Path,
    train_config: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.unlink(missing_ok=True)

    source_copy = output_dir / "source.ipynb"
    executed_notebook = output_dir / "executed.ipynb"
    shutil.copy2(notebook_path, source_copy)

    environment = os.environ.copy()
    environment.update(
        {
            "DATASET_DIR": str(dataset_dir),
            "SEED": str(train_config["seed"]),
            "BATCH_SIZE": str(train_config["batch_size"]),
            "NUM_EPOCHS": str(train_config["epochs"]),
            "NUM_WORKERS": str(train_config["num_workers"]),
            "VAL_RATIO_FROM_TRAIN": str(train_config["val_ratio"]),
            "MLOPS_METRICS_PATH": str(metrics_path),
            "MPLBACKEND": "Agg",
        }
    )

    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        source_copy.name,
        "--output",
        executed_notebook.stem,
        "--ExecutePreprocessor.timeout=-1",
        f"--ExecutePreprocessor.kernel_name={train_config['kernel_name']}",
    ]
    subprocess.run(command, cwd=output_dir, env=environment, check=True)
    source_copy.unlink(missing_ok=True)

    if not executed_notebook.exists():
        raise RuntimeError(f"Executed notebook was not created: {executed_notebook}")
    if not metrics_path.exists():
        raise RuntimeError(f"Notebook did not create its MLOps metrics contract: {metrics_path}")
    return executed_notebook


def enrich_metrics(
    metrics_path: Path,
    repo_root: Path,
    output_dir: Path,
    config: dict[str, Any],
    dataset_summary: dict[str, Any],
) -> dict[str, Any]:
    with metrics_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    data_version = read_data_version(repo_root, Path(dataset_summary["dataset_dir"]))
    git_sha = get_git_sha(repo_root)
    train_config = config["train"]
    eval_version = evaluation_version(
        data_version,
        int(train_config["seed"]),
        float(train_config["val_ratio"]),
    )

    for model in payload["models"]:
        model["checkpoint"] = str(
            (output_dir / model["checkpoint"]).relative_to(repo_root).as_posix()
        )
    selected_name = payload["candidate"]["model_name"]
    payload["candidate"] = next(
        model.copy() for model in payload["models"] if model["model_name"] == selected_name
    )
    payload.update(
        {
            "git_sha": git_sha,
            "data_version": data_version,
            "evaluation_version": eval_version,
            "dataset": dataset_summary,
            "train_params": {
                key: train_config[key]
                for key in ("seed", "batch_size", "epochs", "num_workers", "val_ratio")
            },
        }
    )
    write_json(metrics_path, payload)
    return payload


def normalized_tracking_uri(repo_root: Path, tracking_uri: str) -> str:
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        return tracking_uri

    database = Path(tracking_uri[len(prefix) :]).expanduser()
    if database.is_absolute():
        return tracking_uri
    return prefix + (repo_root / database).resolve().as_posix()


def log_mlflow_run(
    payload: dict[str, Any],
    metrics_path: Path,
    output_dir: Path,
    repo_root: Path,
    mlflow_config: dict[str, Any],
) -> dict[str, Any]:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is enabled but not installed. Run: pip install -r requirements-mlops.txt"
        ) from exc

    tracking_uri = normalized_tracking_uri(repo_root, str(mlflow_config["tracking_uri"]))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(str(mlflow_config["experiment_name"]))

    candidate = payload["candidate"]
    with mlflow.start_run(run_name=f"{candidate['model_name']} candidate") as run:
        mlflow.log_params(payload["train_params"])
        mlflow.log_param("candidate_model", candidate["model_name"])
        mlflow.log_param("data_version", payload["data_version"])
        mlflow.log_param("evaluation_version", payload["evaluation_version"])

        for model in payload["models"]:
            prefix = model["model_name"].lower().replace("-", "_").replace(".", "_")
            mlflow.log_metric(f"{prefix}.best_val_accuracy", model["best_val_accuracy"])
            mlflow.log_metric(f"{prefix}.test_accuracy", model["test_accuracy"])
            mlflow.log_metric(f"{prefix}.test_macro_f1", model["test_macro_f1"])

        mlflow.log_metric("candidate.best_val_accuracy", candidate["best_val_accuracy"])
        mlflow.set_tags(
            {
                "git_sha": payload["git_sha"],
                "model_status": "challenger",
                "candidate_model": candidate["model_name"],
            }
        )

        checkpoint = repo_root / candidate["checkpoint"]
        mlflow.log_artifact(str(checkpoint), artifact_path="model")
        for artifact_name in (
            "executed.ipynb",
            "model_comparison_summary.csv",
            "training_curves.png",
            "confusion_matrices.png",
        ):
            artifact = output_dir / artifact_name
            if artifact.exists():
                mlflow.log_artifact(str(artifact), artifact_path="results")

        payload["mlflow_run_id"] = run.info.run_id
        payload["candidate"]["mlflow_run_id"] = run.info.run_id
        payload["candidate"]["model_uri"] = (
            f"runs:/{run.info.run_id}/model/{checkpoint.name}"
        )
        write_json(metrics_path, payload)
        mlflow.log_artifact(str(metrics_path), artifact_path="metadata")

    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the existing notebook as an MLOps job")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--dataset", help="Override data.path")
    parser.add_argument("--epochs", type=int, help="Override train.epochs")
    parser.add_argument("--batch-size", type=int, help="Override train.batch_size")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = resolve_from_root(repo_root, args.config)
    config = load_config(config_path)

    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size

    dataset_value = args.dataset or config["data"]["path"]
    dataset_dir = resolve_from_root(repo_root, str(dataset_value))
    output_dir = resolve_from_root(repo_root, str(config["output"]["directory"]))
    metrics_path = resolve_from_root(repo_root, str(config["output"]["candidate_metrics"]))

    dataset_summary = validate_dataset(
        dataset_dir, list(config["data"]["expected_classes"])
    )
    execute_notebook(
        repo_root / "coffee_leaf_experiment.ipynb",
        output_dir,
        metrics_path,
        dataset_dir,
        config["train"],
    )
    payload = enrich_metrics(
        metrics_path, repo_root, output_dir, config, dataset_summary
    )

    if bool(config["mlflow"]["enabled"]) and not args.no_mlflow:
        payload = log_mlflow_run(
            payload, metrics_path, output_dir, repo_root, config["mlflow"]
        )

    candidate = payload["candidate"]
    print(
        json.dumps(
            {
                "candidate_model": candidate["model_name"],
                "best_val_accuracy": candidate["best_val_accuracy"],
                "mlflow_run_id": payload.get("mlflow_run_id"),
                "metrics": str(metrics_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
