"""Run the RAG regression evaluation suite and write a snapshot.

Run from the repository root:
    python -m evals.run_suite
    python -m evals.run_suite --baseline
    python -m evals.run_suite --label "retrieval threshold adjustment" --full

This project uses ``dummy_chat.dummy_kb_retrieve.retrieve_kb_chunks`` and
``dummy_chat.dummy_kb_answer.answer_from_kb`` (as in eval_application.py). The
suite intentionally does not import the unrelated ``src.*`` modules.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evals.harness import print_summary
from evals.metric_registry import rule_for

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = "baselines/baseline.json"
CANDIDATE_PATH = "baselines/candidate.json"


def _slug(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def flatten_nested(namespace: str, summary: dict) -> dict[str, Any]:
    """Flatten {metric: {stat: value}} summaries into dotted metric IDs."""
    output: dict[str, Any] = {}
    for metric, stats in summary.items():
        if isinstance(stats, dict):
            for stat, value in stats.items():
                output[f"{namespace}.{_slug(str(metric))}.{stat}"] = value
        else:
            output[f"{namespace}.{_slug(str(metric))}"] = stats
    return output


def prefix_flat(namespace: str, flat: dict) -> dict[str, Any]:
    return {f"{namespace}.{key}": value for key, value in flat.items()}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _prompt_hash() -> str:
    """Hash the prompt/function used by this repository, not a nonexistent src module."""
    try:
        from dummy_chat.dummy_kb_answer import answer_from_kb
        prompt_value = getattr(answer_from_kb, "__doc__", "") or inspect.getsource(answer_from_kb)
    except Exception:
        return "unknown"
    return hashlib.sha256(str(prompt_value).encode("utf-8")).hexdigest()[:12]


def build_metadata(label: str, elapsed: float, full: bool, metric_count: int) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "prompt_hash": _prompt_hash(),
        "label": label,
        "suite_seconds": round(elapsed, 1),
        "full": full,
        "n_metrics": metric_count,
        "judge_model": os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b"),
        "generator_model": os.getenv("GENERATOR_MODEL", "gpt-4o-mini"),
    }


def _invoke_run(module, *preferred_args):
    """Call run() with only supported positional arguments; don't hide run-time errors."""
    run_fn = getattr(module, "run", None)
    if not callable(run_fn):
        raise AttributeError(f"{module.__name__} has no callable run()")
    signature = inspect.signature(run_fn)
    params = list(signature.parameters.values())
    accepts_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    positional = [p for p in params if p.kind in (
        inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD
    )]
    required = [p for p in positional if p.default is inspect.Parameter.empty]
    if not accepts_varargs and len(required) > len(preferred_args):
        raise TypeError(
            f"{module.__name__}.run() requires {len(required)} positional argument(s), "
            f"but the suite has {len(preferred_args)} available. Update that evaluation's "
            "run() to use the live project functions or pass its required dependency explicitly."
        )
    if accepts_varargs:
        return run_fn(*preferred_args)
    return run_fn(*preferred_args[:len(positional)])


def _run_quality(module, args=()):
    summary = _invoke_run(module, *args)
    if not isinstance(summary, dict):
        raise TypeError(f"{module.__name__}.run() must return a dictionary summary; got {type(summary).__name__}.")
    print_summary(module.__name__.split(".")[-1], summary)
    return summary


def _import_optional(module_name: str):
    """Return None only if the requested evaluation module itself is absent."""
    try:
        return __import__(module_name, fromlist=["*"])
    except ModuleNotFoundError as exc:
        if exc.name == module_name or module_name.startswith(f"{exc.name}."):
            return None
        raise


def run_suite(label: str = "", quiet: bool = False, full: bool = False) -> dict:
    started = time.perf_counter()
    metrics: dict[str, Any] = {}

    print("Loading evaluation modules...")
    # These evaluations use the actual dummy_chat retriever/generator flow.
    from evals import eval_retriever, eval_generator, eval_rag_pipeline
    eval_application = _import_optional("evals.eval_application")
    eval_safety = _import_optional("evals.eval_safety")
    eval_ops = _import_optional("evals.eval_ops")

    print("\n[1] Retriever evaluation...")
    metrics.update(flatten_nested("retriever", _run_quality(eval_retriever)))

    print("\n[2] Generator evaluation...")
    metrics.update(flatten_nested("generator", _run_quality(eval_generator)))

    print("\n[3] End-to-end RAG pipeline evaluation (Groq judge)...")
    metrics.update(flatten_nested("pipeline", _run_quality(eval_rag_pipeline)))

    if eval_application is not None:
        print("\n[4] Application evaluation (live retrieve_kb_chunks -> answer_from_kb)...")
        metrics.update(flatten_nested("application", _run_quality(eval_application)))

    if eval_safety is not None:
        print("\n[5] Safety evaluation...")
        safety_fn = getattr(eval_safety, "run_safety", None)
        if callable(safety_fn):
            summary = safety_fn(verbose=not quiet)
            if not isinstance(summary, dict):
                raise TypeError("eval_safety.run_safety() must return a dictionary summary.")
            metrics.update(prefix_flat("safety", summary))
        else:
            metrics.update(flatten_nested("safety", _run_quality(eval_safety)))

    if eval_ops is not None:
        print("\n[6] Operational evaluation...")
        ops_fn = getattr(eval_ops, "run_ops", None)
        if callable(ops_fn):
            summary = ops_fn(verbose=not quiet)
            if not isinstance(summary, dict):
                raise TypeError("eval_ops.run_ops() must return a dictionary summary.")
            metrics.update(prefix_flat("ops", summary))
        else:
            metrics.update(flatten_nested("ops", _run_quality(eval_ops)))

    elapsed = time.perf_counter() - started
    if not full:
        metrics = {key: value for key, value in metrics.items() if rule_for(key)["kind"] != "info"}
    return {"metadata": build_metadata(label, elapsed, full, len(metrics)), "metrics": metrics}


def write_snapshot(snapshot: dict, path: str) -> str:
    destination = Path(path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(snapshot, file, indent=2, sort_keys=True, allow_nan=True)
    return str(destination)


def print_snapshot(snapshot: dict) -> None:
    metadata = snapshot["metadata"]
    print("\n" + "=" * 76)
    print("REGRESSION SNAPSHOT")
    print("=" * 76)
    print(f"label       : {metadata.get('label') or '(none)'}")
    print(f"created_at  : {metadata['created_at']}")
    print(f"git_sha     : {metadata['git_sha']}    prompt_hash: {metadata['prompt_hash']}")
    print(f"suite_time  : {metadata['suite_seconds']}s    metrics: {metadata['n_metrics']}")
    print(f"judge_model : {metadata['judge_model']}")
    print("-" * 76)
    for key, value in sorted(snapshot["metrics"].items()):
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {key:<54} {shown}")
    print("=" * 76)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete RAG regression evaluation suite.")
    parser.add_argument("--baseline", action="store_true", help=f"write baseline at {BASELINE_PATH}")
    parser.add_argument("--out", default=None, help="custom output snapshot path")
    parser.add_argument("--label", default="", help="describe the code/config change being evaluated")
    parser.add_argument("--quiet", action="store_true", help="reduce output where supported")
    parser.add_argument("--full", action="store_true", help="include informational metrics in the snapshot")
    args = parser.parse_args()
    output_path = args.out or (BASELINE_PATH if args.baseline else CANDIDATE_PATH)
    snapshot = run_suite(label=args.label, quiet=args.quiet, full=args.full)
    print_snapshot(snapshot)
    print(f"\nWrote snapshot -> {write_snapshot(snapshot, output_path)}")


if __name__ == "__main__":
    main()
