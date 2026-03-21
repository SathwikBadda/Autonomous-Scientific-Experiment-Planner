"""
example_run.py - Demonstrates the pipeline with two example inputs.
Run this AFTER configuring .env to validate your setup end-to-end.
"""
import json
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from agents.workflow import run_pipeline
from utils.logger import setup_logging
from utils.output_formatter import format_final_output

setup_logging()


EXAMPLES = [
    # ── Example 1: Plain text input ──────────────────────────────────────
    "Improve transformer efficiency for long context language modeling",

    # ── Example 2: Structured input ───────────────────────────────────────
    {
        "domain": "NLP",
        "task": "Text Generation",
        "constraint": "Low compute budget (single GPU)",
    },
]


def run_example(research_input, label: str):
    print(f"\n{'='*70}")
    print(f"  EXAMPLE: {label}")
    print(f"  Input: {json.dumps(research_input) if isinstance(research_input, dict) else research_input}")
    print(f"{'='*70}\n")

    try:
        state = run_pipeline(research_input)
        output = format_final_output(state)

        print("📋 PROBLEM STATEMENT:")
        print(f"  {output['problem_statement']}\n")

        print("📚 LITERATURE SUMMARY (excerpt):")
        print(f"  {output['literature_summary'][:400]}...\n")

        print("🔍 IDENTIFIED GAPS:")
        for i, gap in enumerate(output["identified_gaps"][:3], 1):
            print(f"  {i}. {gap}")

        print("\n💡 HYPOTHESES:")
        for i, h in enumerate(output["hypotheses"][:2], 1):
            print(f"  {i}. {h[:200]}...")

        print("\n🧪 EXPERIMENT PLAN:")
        plan = output["experiment_plan"]
        print(f"  Architecture: {plan.get('model_architecture', 'N/A')[:150]}")
        print(f"  Training:     {str(plan.get('training_strategy', 'N/A'))[:150]}")
        print(f"  Baselines:    {', '.join(plan.get('baseline_models', [])[:3])}")
        print(f"  Metrics:      {', '.join(plan.get('evaluation_metrics', [])[:4])}")

        print("\n📦 DATASETS:")
        for d in output["datasets"][:4]:
            print(f"  • {d}")

        print(f"\n🎯 EXPECTED OUTCOMES:")
        print(f"  {output['expected_outcomes'][:300]}")

        print(f"\n📊 SCORES:")
        print(f"  Novelty:      {output['novelty_score']:.1f}/10")
        print(f"  Feasibility:  {output['feasibility_score']:.1f}/10")
        print(f"  Impact:       {output['impact_score']:.1f}/10")
        print(f"  Composite:    {output['composite_score']:.2f}/10")
        print(f"  Verdict:      {output['overall_recommendation'].upper()}")

        print(f"\n🔬 CRITIQUE:")
        print(f"  {output['critique_summary']}")

        print(f"\n🔗 Pipeline Trace: {' → '.join(output['agent_trace'])}")
        print(f"📄 Papers Retrieved: {output['papers_retrieved']}")
        if output.get("trace_id"):
            print(f"📡 Langfuse Trace ID: {output['trace_id']}")

        # Save full JSON output
        out_path = Path(f"output_example_{label.replace(' ', '_')}.json")
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"\n✅ Full output saved to: {out_path}")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    run_example(EXAMPLES[0], "Plain Text Input")
    # Uncomment to run second example:
    # run_example(EXAMPLES[1], "Structured Input")
