"""Aggregate prediction statistics for dashboard cards and API summary."""

from __future__ import annotations

from typing import Any, Sequence


def build_summary(
    leaves: Sequence[dict[str, Any]],
    healthy_label: str,
    *,
    disease_classes: Sequence[str] | None = None,
    mode: str | None = None,
    fallback_to_single_leaf: bool = False,
) -> dict[str, Any]:
    total = len(leaves)
    healthy = 0
    diseased = 0
    multi_disease = 0
    confidences: list[float] = []
    class_counts: dict[str, int] = {}
    detected_diseases: set[str] = set()
    diseases = list(disease_classes or [])
    if not diseases:
        for leaf in leaves:
            for label in leaf.get("predicted_labels", []):
                if label != healthy_label and label not in diseases:
                    diseases.append(label)
    disease_counts = {name: 0 for name in diseases}

    for leaf in leaves:
        labels = list(leaf.get("predicted_labels", []))
        confidences.append(float(leaf.get("classification_confidence", 0.0)))
        disease_labels = [name for name in labels if name != healthy_label]
        is_healthy = not disease_labels
        if is_healthy:
            healthy += 1
        else:
            diseased += 1
            if len(disease_labels) >= 2:
                multi_disease += 1
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
            if label != healthy_label:
                detected_diseases.add(label)
                if label in disease_counts:
                    disease_counts[label] += 1
                else:
                    disease_counts[label] = 1

    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    summary: dict[str, Any] = {
        "total_leaves": total,
        "healthy_leaves": healthy,
        "diseased_leaves": diseased,
        "average_confidence": average_confidence,
        "detected_diseases": sorted(detected_diseases),
        "class_counts": class_counts,
        "disease_counts": disease_counts,
        "multi_disease_leaves": multi_disease,
    }
    if mode is not None:
        summary["mode"] = mode
        summary["fallback_to_single_leaf"] = bool(fallback_to_single_leaf)
    return summary


def format_leaf_prediction_label(labels: Sequence[str], healthy_label: str) -> str:
    if not labels:
        return healthy_label
    disease_labels = [name for name in labels if name != healthy_label]
    if not disease_labels:
        return healthy_label
    return " + ".join(disease_labels)


def display_label_for_leaf(leaf: dict[str, Any], healthy_label: str) -> str:
    labels = list(leaf.get("predicted_labels", leaf.get("labels", [])))
    return format_leaf_prediction_label(labels, healthy_label)


def is_healthy_leaf(leaf: dict[str, Any], healthy_label: str) -> bool:
    labels = list(leaf.get("predicted_labels", leaf.get("labels", [])))
    return not any(name != healthy_label for name in labels)
