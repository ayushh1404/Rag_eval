"""Shared helpers for the RAG evaluation suite.

Provides flexible golden JSON loading and per-metric DeepEval summaries.
Works across DeepEval versions that expose result.test_results or return a list.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_goldens(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Golden dataset not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = next(
            (payload[key] for key in ("goldens", "records", "examples", "data", "test_cases", "items")
             if isinstance(payload.get(key), list)),
            None,
        )
        if records is None and any(key in payload for key in ("query", "input", "question", "user_query")):
            records = [payload]
        if records is None:
            raise ValueError("Unsupported golden JSON structure: expected a list of records.")
    else:
        raise ValueError("Golden dataset must be a JSON list or object.")

    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError("Golden dataset is empty or contains non-object records.")
    return records


def get_query(record: dict[str, Any], number: int = 1) -> str:
    for key in ("query", "input", "question", "user_query"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Record {number} has no query/input/question field.")


def summarize_by_metric(result) -> dict[str, dict[str, Any]]:
    """Convert a DeepEval EvaluationResult into metric-level score statistics."""
    test_results = getattr(result, "test_results", None)
    if test_results is None:
        test_results = result if isinstance(result, list) else []

    buckets: dict[str, dict[str, Any]] = {}
    for test_result in test_results:
        metrics = getattr(test_result, "metrics_data", None)
        if metrics is None:
            metrics = getattr(test_result, "metrics", None)
        for metric in metrics or []:
            name = getattr(metric, "name", None) or getattr(metric, "__name__", "unknown")
            bucket = buckets.setdefault(name, {"scores": [], "passed": 0, "total": 0})
            bucket["total"] += 1
            score = getattr(metric, "score", None)
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                bucket["scores"].append(float(score))
            if getattr(metric, "success", False) is True:
                bucket["passed"] += 1

    summary = {}
    for name, bucket in buckets.items():
        scores = bucket["scores"]
        summary[name] = {
            "n": bucket["total"],
            "pass_rate": 100.0 * bucket["passed"] / bucket["total"] if bucket["total"] else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else float("nan"),
            "min_score": min(scores) if scores else float("nan"),
            "max_score": max(scores) if scores else float("nan"),
        }
    return summary


def print_summary(title: str, summary: dict[str, dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print(f"{title} (per-metric summary)")
    print("=" * 60)
    for name, stats in sorted(summary.items()):
        avg = f"{stats['avg_score']:.2f}" if not math.isnan(stats["avg_score"]) else "nan"
        print(f"  {name:<30} pass_rate={stats['pass_rate']:5.0f}%  avg={avg}  n={stats['n']}")
    print("=" * 60)
