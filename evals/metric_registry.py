"""Rules used by the RAG regression comparator.

A metric rule defines its good direction, verdict impact, and noise tolerance.
Tolerances are absolute unless ``rel_tol`` is non-zero; the comparator should
classify a regression only when the worsening exceeds max(tol, rel_tol*abs(baseline)).
"""

GATE_HIGHER_AVG = {"direction": "higher", "kind": "gate", "tol": 0.02, "rel_tol": 0.0}
GATE_LOWER_AVG = {"direction": "lower", "kind": "gate", "tol": 0.02, "rel_tol": 0.0}
QUALITY_GUARD = {"direction": "higher", "kind": "guardrail", "tol": 0.05, "rel_tol": 0.0}
LATENCY_GUARD = {"direction": "lower", "kind": "guardrail", "tol": 0.0, "rel_tol": 0.25}
COST_GUARD = {"direction": "lower", "kind": "guardrail", "tol": 0.0, "rel_tol": 0.15}
SUCCESS_GUARD = {"direction": "higher", "kind": "guardrail", "tol": 1.0, "rel_tol": 0.0}
ERROR_GUARD = {"direction": "lower", "kind": "guardrail", "tol": 1.0, "rel_tol": 0.0}
SLO_BOOL_GUARD = {"direction": "higher", "kind": "guardrail", "tol": 0.0, "rel_tol": 0.0, "bool": True}
INFO = {"direction": "higher", "kind": "info", "tol": 0.0, "rel_tol": 0.0}

LATENCY_GUARDED = {"ops.latency.e2e_p95_ms"}


def rule_for(metric_id: str) -> dict:
    """Return the comparison rule for a flattened metric identifier."""
    mid = str(metric_id).strip().lower()

    # Safety metrics are release gates, evaluated on average scores.
    if mid == "safety.toxicity.avg_toxicity" or mid.endswith("toxicity.avg_toxicity"):
        return GATE_LOWER_AVG
    if mid.startswith("safety.") and mid.endswith("avg_score"):
        return GATE_HIGHER_AVG

    # Other quality average scores are guardrails.
    if mid.endswith("avg_score"):
        return QUALITY_GUARD

    # Operational guardrails.
    if mid in LATENCY_GUARDED:
        return LATENCY_GUARD
    if mid == "ops.cost.cost_per_query_usd":
        return COST_GUARD
    if mid == "ops.reliability.success_rate":
        return SUCCESS_GUARD
    if mid == "ops.reliability.error_rate":
        return ERROR_GUARD
    if mid.endswith("_pass") or mid.endswith(".slo_pass") or mid.endswith(".budget_pass"):
        return SLO_BOOL_GUARD

    # Pass rates, distributions, counts, and unrecognized metrics are informational.
    return INFO
