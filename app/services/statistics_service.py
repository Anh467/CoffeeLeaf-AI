"""Aggregate prediction statistics for dashboard cards and API summary."""

from __future__ import annotations

from typing import Any, Sequence


def build_summary(
    leaves: Sequence[dict[str, Any]],
    healthy_label: str,
) -> dict[str, Any]:
    total = len(leaves)
    healthy = 0
    diseased = 0
    confidences: list[float] = []
    class_counts: dict[str, int] = {}
    detected_diseases: set[str] = set()

    for leaf in leaves:
        labels = list(leaf.get("predicted_labels", []))
        confidences.append(float(leaf.get("classification_confidence", 0.0)))
        is_healthy = labels == [healthy_label] or (
            len(labels) == 1 and labels[0] == healthy_label
        )
        if is_healthy:
            healthy += 1
        else:
            diseased += 1
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
            if label != healthy_label:
                detected_diseases.add(label)

    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return {
        "total_leaves": total,
        "healthy_leaves": healthy,
        "diseased_leaves": diseased,
        "average_confidence": average_confidence,
        "detected_diseases": sorted(detected_diseases),
        "class_counts": class_counts,
    }


def format_leaf_prediction_label(labels: Sequence[str], healthy_label: str) -> str:
    if not labels:
        return healthy_label
    if labels == [healthy_label]:
        return healthy_label.capitalize()
    return " + ".join(name.capitalize() for name in labels)
