"""
utils/output_formatter.py - Converts final pipeline state into strict JSON output.
"""
from __future__ import annotations

from typing import Any


def format_final_output(state: dict) -> dict:
    """
    Maps the raw pipeline state to the strict output JSON schema.
    All fields are guaranteed to be present (with defaults if missing).
    """

    # ── experiment_plan normalisation ──────────────────────────────────────
    raw_plan = state.get("experiment_plan", {})

    # Handle nested vs flat experiment_plan structures
    if "experiment_plan" in raw_plan:
        plan = raw_plan["experiment_plan"]
    else:
        plan = raw_plan

    training = plan.get("training_strategy", {})
    if isinstance(training, str):
        training = {"description": training}

    baseline_models: list[str] = []
    for b in plan.get("baseline_models", []):
        if isinstance(b, dict):
            baseline_models.append(b.get("name", str(b)))
        else:
            baseline_models.append(str(b))

    # ── datasets normalisation ─────────────────────────────────────────────
    raw_datasets = state.get("datasets", [])
    dataset_names: list[str] = []
    for d in raw_datasets:
        if isinstance(d, dict):
            dataset_names.append(d.get("name", str(d)))
        else:
            dataset_names.append(str(d))

    # ── gaps & hypotheses ──────────────────────────────────────────────────
    raw_gaps = state.get("identified_gaps", [])
    gap_strings: list[str] = []
    for g in raw_gaps:
        if isinstance(g, dict):
            gap_strings.append(g.get("gap", str(g)))
        else:
            gap_strings.append(str(g))

    raw_hyps = state.get("hypotheses", [])
    hyp_strings: list[str] = []
    for h in raw_hyps:
        if isinstance(h, dict):
            hyp_strings.append(h.get("hypothesis", str(h)))
        else:
            hyp_strings.append(str(h))

    # ── scores ─────────────────────────────────────────────────────────────
    novelty = _clamp(state.get("novelty_score", 0))
    feasibility = _clamp(state.get("feasibility_score", 0))
    impact = _clamp(state.get("impact_score", 0))

    return {
        "problem_statement": state.get("problem_statement", ""),
        "literature_summary": state.get("literature_summary", ""),
        "identified_gaps": gap_strings,
        "hypotheses": hyp_strings,
        "experiment_plan": {
            "model_architecture": plan.get("model_architecture", ""),
            "training_strategy": _stringify(training),
            "baseline_models": baseline_models,
            "evaluation_metrics": plan.get("evaluation_metrics", []),
        },
        "datasets": dataset_names,
        "expected_outcomes": state.get("expected_outcomes", ""),
        "novelty_score": novelty,
        "feasibility_score": feasibility,
        "impact_score": impact,
        # ── extended fields (bonus) ──────────────────────────────────────
        "composite_score": state.get("composite_score", 0),
        "overall_recommendation": state.get("overall_recommendation", "revise"),
        "critique_summary": _extract_critique_summary(state.get("critique", {})),
        "agent_trace": state.get("agent_trace", []),
        "papers_retrieved": state.get("total_papers_fetched", 0),
        "trace_id": state.get("trace_id"),
        "error": state.get("error"),
    }


def _clamp(val: Any) -> float:
    try:
        return max(0.0, min(10.0, float(val)))
    except (TypeError, ValueError):
        return 0.0


def _stringify(val: Any) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return "; ".join(f"{k}: {v}" for k, v in val.items())
    return str(val)


def _extract_critique_summary(critique: dict) -> str:
    if not critique:
        return ""
    strengths = critique.get("strengths", [])
    weaknesses = [w.get("issue", str(w)) if isinstance(w, dict) else str(w)
                  for w in critique.get("weaknesses", [])]
    parts = []
    if strengths:
        parts.append("Strengths: " + "; ".join(strengths[:2]))
    if weaknesses:
        parts.append("Weaknesses: " + "; ".join(weaknesses[:2]))
    return " | ".join(parts)
