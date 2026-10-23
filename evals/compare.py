"""Compare baseline and candidate evaluation snapshots.

Exit codes: PASS=0, FAIL=1, REVIEW=2.
Run from project root:
    python -m evals.compare
    python -m evals.compare --baseline baselines/baseline.json --candidate baselines/candidate.json --all
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from evals.metric_registry import rule_for

BASELINE_PATH = "baselines/baseline.json"
CANDIDATE_PATH = "baselines/candidate.json"
EXIT_CODE = {"PASS": 0, "FAIL": 1, "REVIEW": 2}


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Snapshot must contain a JSON object: {path}")
    if not isinstance(payload.get("metrics", {}), dict):
        raise ValueError(f"Snapshot 'metrics' must be a JSON object: {path}")
    return payload


def _is_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not (isinstance(value, float) and math.isnan(value))
    )


def classify(metric_id, baseline_value, candidate_value, rule):
    row = {
        "id": metric_id, "baseline": baseline_value, "candidate": candidate_value,
        "kind": rule["kind"], "direction": rule["direction"],
        "delta": None, "status": None,
    }
    if baseline_value is None:
        row["status"] = "new"
        return row
    if candidate_value is None:
        row["status"] = "dropped"
        return row

    if rule["kind"] == "info":
        if _is_number(baseline_value) and _is_number(candidate_value):
            row["delta"] = candidate_value - baseline_value
        row["status"] = "info"
        return row

    if rule.get("bool") or isinstance(baseline_value, bool) or isinstance(candidate_value, bool):
        if baseline_value == candidate_value:
            row["status"] = "flat"
        elif (not baseline_value) and candidate_value:
            row["status"] = "improved"
        else:
            row["status"] = "blocked" if rule["kind"] == "gate" else "regressed"
        return row

    if not (_is_number(baseline_value) and _is_number(candidate_value)):
        row["status"] = "info"
        return row

    delta = candidate_value - baseline_value
    row["delta"] = delta
    improved = delta > 0 if rule["direction"] == "higher" else delta < 0
    if improved:
        row["status"] = "improved"
        return row

    tolerance = max(float(rule.get("tol", 0)), float(rule.get("rel_tol", 0)) * abs(baseline_value))
    row["status"] = "flat" if abs(delta) <= tolerance else (
        "blocked" if rule["kind"] == "gate" else "regressed"
    )
    return row


def compare(baseline: dict, candidate: dict):
    baseline_metrics = baseline.get("metrics", {})
    candidate_metrics = candidate.get("metrics", {})
    metric_ids = sorted(set(baseline_metrics) | set(candidate_metrics))
    rows = [
        classify(mid, baseline_metrics.get(mid), candidate_metrics.get(mid), rule_for(mid))
        for mid in metric_ids
    ]
    if any(row["status"] == "blocked" for row in rows):
        verdict = "FAIL"
    elif any(row["status"] == "regressed" for row in rows):
        verdict = "REVIEW"
    else:
        verdict = "PASS"
    return verdict, rows


def _fmt(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.4g}"
    return str(value)


def print_report(verdict, rows, show_all=False):
    order = {"blocked": 0, "regressed": 1, "dropped": 2, "new": 3,
             "improved": 4, "flat": 5, "info": 6}
    shown = [row for row in rows if show_all or row["kind"] != "info"]
    shown.sort(key=lambda row: (order.get(row["status"], 9), row["id"]))
    print("=" * 96)
    print(f"{'metric':<46} {'baseline':>10} {'candidate':>10} {'delta':>10}  status")
    print("-" * 96)
    for row in shown:
        delta = row["delta"]
        delta_text = f"{delta:+.4g}" if _is_number(delta) else ""
        marker = "  <<<" if row["status"] in ("blocked", "regressed") else ""
        print(f"{row['id']:<46} {_fmt(row['baseline']):>10} {_fmt(row['candidate']):>10} "
              f"{delta_text:>10}  {row['status']}{marker}")
    counts = Counter(row["status"] for row in rows)
    summary = "  ".join(
        f"{key}={counts[key]}" for key in
        ("blocked", "regressed", "improved", "flat", "new", "dropped", "info")
        if counts[key]
    )
    print("-" * 96)
    print(summary or "No metrics to compare.")
    if not show_all and counts["info"]:
        print(f"({counts['info']} info metrics hidden; pass --all to show them)")
    print("=" * 96)
    banner = {
        "PASS": "PASS   -- no gate blocked, no guardrail regressed. Safe to promote.",
        "REVIEW": "REVIEW -- a guardrail regressed beyond tolerance. A human decides.",
        "FAIL": "FAIL   -- a gate regressed. Blocked.",
    }[verdict]
    print(f"VERDICT: {banner}")
    print("=" * 96)


def main():
    parser = argparse.ArgumentParser(description="Compare two eval snapshots and return a verdict.")
    parser.add_argument("--baseline", default=BASELINE_PATH)
    parser.add_argument("--candidate", default=CANDIDATE_PATH)
    parser.add_argument("--all", action="store_true", help="also show info metrics")
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    baseline_meta = baseline.get("metadata", {})
    candidate_meta = candidate.get("metadata", {})
    print(f"baseline  : {baseline_meta.get('label') or args.baseline} "
          f"({baseline_meta.get('git_sha', '?')}, prompt {baseline_meta.get('prompt_hash', '?')})")
    print(f"candidate : {candidate_meta.get('label') or args.candidate} "
          f"({candidate_meta.get('git_sha', '?')}, prompt {candidate_meta.get('prompt_hash', '?')})")
    verdict, rows = compare(baseline, candidate)
    print_report(verdict, rows, show_all=args.all)
    sys.exit(EXIT_CODE[verdict])


if __name__ == "__main__":
    main()
