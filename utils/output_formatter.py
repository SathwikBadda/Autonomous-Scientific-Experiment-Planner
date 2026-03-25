"""
utils/output_formatter.py - Converts final pipeline state to strict JSON output.
All fields guaranteed to be present with defaults. Extended with model recommendations,
dataset URLs, resources, and literature summary bullets.
"""
from __future__ import annotations
from typing import Any


def format_final_output(state: dict) -> dict:
    # ── experiment_plan normalisation ──────────────────────────────────────
    raw_plan = state.get("experiment_plan", {})
    if "experiment_plan" in raw_plan:
        plan = raw_plan["experiment_plan"]
    else:
        plan = raw_plan

    training = plan.get("training_strategy", {})
    if isinstance(training, str):
        training = {"description": training}

    baseline_models: list = []
    for b in plan.get("baseline_models", []):
        if isinstance(b, dict):
            baseline_models.append(b.get("name", str(b)))
        else:
            baseline_models.append(str(b))

    # ── dataset normalisation ─────────────────────────────────────────────
    raw_datasets = state.get("datasets", [])
    datasets_rich: list = []
    dataset_names: list = []
    for d in raw_datasets:
        if isinstance(d, dict):
            datasets_rich.append(d)
            dataset_names.append(d.get("name", str(d)))
        else:
            dataset_names.append(str(d))
            datasets_rich.append({"name": str(d), "url": "", "description": "", "use_in_experiment": ""})

    # ── gaps ──────────────────────────────────────────────────────────────
    raw_gaps = state.get("identified_gaps", [])
    gap_strings: list = []
    gaps_rich: list = []
    for g in raw_gaps:
        if isinstance(g, dict):
            gap_str = g.get("gap", str(g))
            gap_strings.append(gap_str)
            gaps_rich.append(g)
        else:
            gap_strings.append(str(g))
            gaps_rich.append({"gap": str(g), "gap_type": "general", "evidence": "", "impact": "", "difficulty": "medium"})

    # ── hypotheses ────────────────────────────────────────────────────────
    raw_hyps = state.get("hypotheses", [])
    hyp_strings: list = []
    hypotheses_rich: list = []
    for h in raw_hyps:
        if isinstance(h, dict):
            hyp_str = h.get("hypothesis", str(h))
            hyp_strings.append(hyp_str)
            hypotheses_rich.append(h)
        else:
            hyp_strings.append(str(h))
            hypotheses_rich.append({"hypothesis": str(h), "approach": "", "expected_improvement": "", "confidence": "medium"})

    # ── scores ────────────────────────────────────────────────────────────
    novelty = _clamp(state.get("novelty_score", 0))
    feasibility = _clamp(state.get("feasibility_score", 0))
    impact = _clamp(state.get("impact_score", 0))

    # ── model recommendations ─────────────────────────────────────────────
    recommended_models = plan.get("recommended_models", [])

    # ── resource requirements ─────────────────────────────────────────────
    resource_requirements = plan.get("resource_requirements", {})

    # ── literature summary bullets ────────────────────────────────────────
    paper_analyses = state.get("paper_analyses", [])
    lit_bullets = state.get("literature_summary_bullets", [])
    if not lit_bullets and paper_analyses:
        # Extract from paper analyses if not set directly
        analyses = paper_analyses if isinstance(paper_analyses, list) else []
        if analyses and isinstance(analyses[0], dict):
            lit_bullets = analyses[0].get("literature_summary_bullets", [])

    # ── retrieved papers ──────────────────────────────────────────────────
    retrieved_papers = state.get("retrieved_papers", [])
    top_papers_display = [
        {
            "title": p.get("title", ""),
            "arxiv_id": p.get("arxiv_id", ""),
            "url": p.get("url", ""),
            "year": p.get("year", ""),
            "abstract": p.get("abstract", "")[:300],
        }
        for p in retrieved_papers[:5]
    ]

    # ── critique ──────────────────────────────────────────────────────────
    critique = state.get("critique", {})
    critique_summary = state.get("critique_summary", "") or _extract_critique_summary(critique)

    return {
        # ── core output ──────────────────────────────────────────────────
        "problem_statement": state.get("problem_statement", ""),
        "literature_summary": state.get("literature_summary", ""),
        "literature_summary_bullets": lit_bullets,
        "identified_gaps": gap_strings,
        "identified_gaps_rich": gaps_rich,
        "hypotheses": hyp_strings,
        "hypotheses_rich": hypotheses_rich,
        # ── experiment plan ───────────────────────────────────────────────
        "experiment_plan": {
            "title": plan.get("title", ""),
            "model_architecture": plan.get("model_architecture", ""),
            "key_components": plan.get("key_components", []),
            "training_strategy": _stringify(training),
            "training_strategy_dict": training if isinstance(training, dict) else {},
            "experiment_steps": plan.get("experiment_steps", []),
            "baseline_models": baseline_models,
            "evaluation_metrics": plan.get("evaluation_metrics", []),
            "ablation_studies": plan.get("ablation_studies", []),
            "success_criteria": plan.get("success_criteria", ""),
            "estimated_compute": plan.get("estimated_compute", ""),
            "timeline_weeks": plan.get("timeline_weeks", 0),
        },
        "recommended_models": recommended_models,
        "resource_requirements": resource_requirements,
        # ── datasets ──────────────────────────────────────────────────────
        "datasets": dataset_names,
        "datasets_rich": datasets_rich,
        # ── outputs & scores ──────────────────────────────────────────────
        "expected_outcomes": state.get("expected_outcomes", ""),
        "novelty_score": novelty,
        "feasibility_score": feasibility,
        "impact_score": impact,
        "composite_score": state.get("composite_score", 0),
        "overall_recommendation": state.get("overall_recommendation", "revise"),
        "critique_summary": critique_summary,
        "critique": critique,
        # ── architecture & outline ────────────────────────────────────────
        "architecture_diagram": state.get("architecture_diagram", ""),
        "research_paper_outline": state.get("research_paper_outline", ""),
        # ── meta ──────────────────────────────────────────────────────────
        "agent_trace": state.get("agent_trace", []),
        "papers_retrieved": state.get("total_papers_fetched", len(retrieved_papers)),
        "top_papers": top_papers_display,
        "proposed_improvements": state.get("proposed_improvements", []),
        "gap_summary": state.get("gap_summary", ""),
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
        parts = []
        for k, v in val.items():
            parts.append(f"**{k.replace('_', ' ').title()}:** {v}")
        return " | ".join(parts)
    return str(val)


def _extract_critique_summary(critique: dict) -> str:
    if not critique:
        return ""
    strengths = critique.get("strengths", [])
    weaknesses = [
        w.get("issue", str(w)) if isinstance(w, dict) else str(w)
        for w in critique.get("weaknesses", [])
    ]
    parts = []
    if strengths:
        parts.append("Strengths: " + "; ".join(str(s) for s in strengths[:2]))
    if weaknesses:
        parts.append("Weaknesses: " + "; ".join(weaknesses[:2]))
    return " | ".join(parts)
