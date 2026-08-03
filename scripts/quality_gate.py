from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def evaluate_quality_gate(
    candidate_payload: dict[str, Any],
    champion_payload: dict[str, Any],
    gate_config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    candidate = candidate_payload["candidate"]
    candidate_score = float(candidate["best_val_accuracy"])
    champion_score = float(champion_payload["best_val_accuracy"])
    minimum_score = float(gate_config["minimum_val_accuracy"])
    minimum_improvement = float(gate_config["minimum_improvement"])
    improvement = candidate_score - champion_score

    checkpoint = repo_root / candidate["checkpoint"]
    same_evaluation = (
        not gate_config.get("require_same_evaluation_version", True)
        or not champion_payload.get("evaluation_version")
        or candidate_payload.get("evaluation_version")
        == champion_payload.get("evaluation_version")
    )
    checks = {
        "minimum_val_accuracy": candidate_score >= minimum_score,
        "strict_improvement": improvement > minimum_improvement,
        "checkpoint_exists": checkpoint.is_file() and checkpoint.stat().st_size > 0,
        "same_evaluation_version": same_evaluation,
    }
    approved = all(checks.values())

    failed_checks = [name for name, passed in checks.items() if not passed]
    reason = (
        f"Candidate improved validation accuracy from {champion_score:.6f} "
        f"to {candidate_score:.6f}."
        if approved
        else f"Rejected by: {', '.join(failed_checks)}."
    )
    return {
        "schema_version": 1,
        "approved": approved,
        "reason": reason,
        "checks": checks,
        "candidate": {
            "model_name": candidate["model_name"],
            "best_val_accuracy": candidate_score,
            "mlflow_run_id": candidate.get("mlflow_run_id"),
            "model_uri": candidate.get("model_uri"),
            "evaluation_version": candidate_payload.get("evaluation_version"),
        },
        "champion": {
            "model_name": champion_payload["model_name"],
            "best_val_accuracy": champion_score,
            "mlflow_run_id": champion_payload.get("mlflow_run_id"),
            "model_uri": champion_payload.get("model_uri"),
            "evaluation_version": champion_payload.get("evaluation_version"),
        },
        "improvement": improvement,
    }


def promote_candidate(
    report: dict[str, Any],
    candidate_payload: dict[str, Any],
    champion_path: Path,
) -> None:
    if not report["approved"]:
        raise ValueError("Cannot promote a candidate that failed the quality gate")

    candidate = candidate_payload["candidate"]
    champion = {
        "schema_version": 1,
        "model_name": candidate["model_name"],
        "best_val_accuracy": candidate["best_val_accuracy"],
        "test_accuracy": candidate.get("test_accuracy"),
        "test_macro_f1": candidate.get("test_macro_f1"),
        "model_uri": candidate.get("model_uri"),
        "mlflow_run_id": candidate.get("mlflow_run_id"),
        "git_sha": candidate_payload.get("git_sha"),
        "data_version": candidate_payload.get("data_version"),
        "evaluation_version": candidate_payload.get("evaluation_version"),
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(champion_path, champion)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a candidate with the champion")
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--candidate")
    parser.add_argument("--champion")
    parser.add_argument("--report")
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Update the champion manifest only when every gate passes",
    )
    return parser.parse_args()


def resolve(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = resolve(repo_root, args.config)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    candidate_path = resolve(
        repo_root, args.candidate or config["output"]["candidate_metrics"]
    )
    champion_path = resolve(
        repo_root, args.champion or config["quality_gate"]["champion_manifest"]
    )
    report_path = resolve(
        repo_root, args.report or config["quality_gate"]["report"]
    )

    candidate_payload = read_json(candidate_path)
    champion_payload = read_json(champion_path)
    report = evaluate_quality_gate(
        candidate_payload, champion_payload, config["quality_gate"], repo_root
    )
    write_json(report_path, report)

    if args.promote:
        promote_candidate(report, candidate_payload, champion_path)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
