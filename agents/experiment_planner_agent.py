"""
agents/experiment_planner_agent.py - Generates a complete experiment plan
with architecture diagram (Mermaid MD), steps, baselines, datasets, and metrics.
"""
from __future__ import annotations

import json

from agents.base_agent import BaseAgent
from utils.logger import get_logger
from utils.tracer import agent_span

logger = get_logger(__name__)


class ExperimentPlannerAgent(BaseAgent):
    agent_name = "experiment_planner"
    prompt_key = "experiment_planner_agent"

    def run(self, state: dict, trace=None) -> dict:
        with agent_span(trace, "ExperimentPlannerAgent"):
            primary_hypothesis = state.get("primary_hypothesis", "")
            hypotheses = state.get("hypotheses", [])
            constraints = state.get("constraints", [])
            problem_statement = state.get("problem_statement", "")
            retrieved_papers = state.get("retrieved_papers", [])
            identified_gaps = state.get("identified_gaps", [])

            # Fallback if hypothesis is empty
            if not primary_hypothesis:
                primary_hypothesis = f"Investigate novel approaches to: {problem_statement}"

            # Provide paper baselines context
            paper_context = "\n".join(
                f"- {p.get('title', '')} (Year: {p.get('year', 'unknown')})"
                for p in retrieved_papers[:5]
            )

            user_prompt = self._get_user_prompt(
                primary_hypothesis=primary_hypothesis,
                hypotheses=json.dumps(hypotheses, indent=2),
                performance_frontier=paper_context or "Several baselines exist in literature.",
                constraints=json.dumps(constraints),
            )

            response = self._call_llm(user_prompt=user_prompt, trace=trace)
            text = self._extract_text(response)

            try:
                parsed = self._parse_json(text)
            except ValueError:
                logger.warning("experiment_planner_parse_fallback", text_preview=text[:300])
                # Build a reasonable fallback from raw text
                domain = state.get("domain", "AI/ML")
                parsed = {
                    "experiment_plan": {
                        "title": f"Experiment Plan: {problem_statement[:60]}",
                        "model_architecture": f"Novel architecture for {domain}",
                        "key_components": ["Feature extraction", "Core model", "Post-processing"],
                        "training_strategy": {
                            "optimizer": "AdamW",
                            "learning_rate": "1e-4 with cosine annealing",
                            "epochs": 100,
                            "batch_size": 32,
                            "regularization": "Dropout(0.3), Weight Decay(1e-5)",
                        },
                        "experiment_steps": [
                            "1. Setup environment and baseline reproduction",
                            "2. Implement proposed architecture modifications",
                            "3. Train on primary dataset with validation splits",
                            "4. Evaluate against baselines on all metrics",
                            "5. Ablation studies on key components",
                            "6. Statistical significance testing",
                        ],
                        "baseline_models": [
                            p.get("title", f"Baseline {i+1}")[:60]
                            for i, p in enumerate(retrieved_papers[:3])
                        ] or ["SOTA Method A", "SOTA Method B", "Classic Baseline"],
                        "evaluation_metrics": ["Accuracy", "F1-Score", "Precision", "Recall"],
                        "ablation_studies": [
                            "Component A only",
                            "Component B only",
                            "Full model without pre-training",
                        ],
                        "success_criteria": "Statistically significant improvement over all baselines",
                        "estimated_compute": "4x A100 GPUs, ~72 hours",
                        "timeline_weeks": 12,
                    },
                    "architecture_diagram": self._generate_architecture_diagram(problem_statement, domain),
                    "research_paper_outline": self._generate_paper_outline(problem_statement, hypotheses, identified_gaps),
                }

            experiment_plan = parsed.get("experiment_plan", parsed)

            # Always ensure architecture diagram
            arch_diagram = parsed.get("architecture_diagram", "")
            if not arch_diagram:
                arch_diagram = self._generate_architecture_diagram(
                    problem_statement, state.get("domain", "AI/ML")
                )

            # Always ensure research paper outline
            paper_outline = parsed.get("research_paper_outline", "")
            if not paper_outline:
                paper_outline = self._generate_paper_outline(
                    problem_statement, hypotheses, identified_gaps
                )

            logger.info(
                "experiment_planner_done",
                baselines=len(experiment_plan.get("baseline_models", [])),
                metrics=len(experiment_plan.get("evaluation_metrics", [])),
                has_diagram=bool(arch_diagram),
            )

            return {
                **state,
                "experiment_plan": experiment_plan,
                "architecture_diagram": arch_diagram,
                "research_paper_outline": paper_outline,
                "agent_trace": state.get("agent_trace", []) + ["ExperimentPlannerAgent"],
            }

    def _generate_architecture_diagram(self, problem_statement: str, domain: str) -> str:
        """Generate a Mermaid markdown architecture diagram based on the domain."""
        return f"""```mermaid
flowchart TD
    A[🎯 Input Data / Problem: {domain}] --> B[📥 Data Preprocessing]
    B --> C[Feature Extraction Module]
    C --> D{{Core Model Architecture}}
    D --> E[Component A: Primary Method]
    D --> F[Component B: Enhancement Module]
    E --> G[Multi-Scale Fusion]
    F --> G
    G --> H[Prediction Head / Output Layer]
    H --> I[📊 Evaluation & Scoring]
    I --> J{{Threshold / Decision}}
    J -->|Pass| K[✅ Final Output]
    J -->|Fail| L[🔁 Refinement Loop]
    L --> D

    style A fill:#4A90D9,color:#fff
    style K fill:#27AE60,color:#fff
    style D fill:#F39C12,color:#fff
    style I fill:#8E44AD,color:#fff
```"""

    def _generate_paper_outline(self, problem_statement: str, hypotheses: list, gaps: list) -> str:
        """Generate a structured research paper outline."""
        hyp_text = hypotheses[0].get("hypothesis", str(hypotheses[0]))[:200] if hypotheses else "TBD"
        gap_text = "\n".join(f"  - {g.get('gap', str(g))[:100]}" for g in gaps[:3]) if gaps else "  - TBD"
        return f"""# Research Paper Outline

## Title
{problem_statement[:100]}

## Abstract
[To be written after experiments]

## 1. Introduction
- Problem motivation
- Research questions
- Contributions (3-5 bullet points)

## 2. Related Work
- Literature review of retrieved papers
- Gap analysis summary

## 3. Methodology
### 3.1 Problem Formulation
### 3.2 Proposed Architecture
### 3.3 Training Strategy

## 4. Experiments
### 4.1 Datasets
### 4.2 Baselines
### 4.3 Main Results
### 4.4 Ablation Studies
### 4.5 Analysis

## 5. Discussion
- Key findings
- Limitations
- Future work

## 6. Conclusion

---
**Primary Hypothesis:** {hyp_text}

**Key Gaps Addressed:**
{gap_text}
"""
