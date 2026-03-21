import streamlit as st
import json
import os
from pathlib import Path
import sys

# Add project root to sys.path so we can import internal modules
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.workflow import run_pipeline
from utils.output_formatter import format_final_output
from config.settings import config
from utils.logger import setup_logging

# Initialize logging for the backend
setup_logging()

st.set_page_config(
    page_title="Autonomous Scientific Experiment Planner", 
    page_icon="🔬", 
    layout="wide"
)

st.title("🔬 Autonomous Scientific Experiment Planner")
st.markdown("> A multi-agent AI system that **thinks like a scientist** — analyzing literature, identifying research gaps, generating hypotheses, and designing rigorous experiment plans.")

st.sidebar.header("Input Type")
input_option = st.sidebar.radio("Choose how to input your research idea:", ["Plain Text", "Structured Input"])

research_input = None

if input_option == "Plain Text":
    st.header("Plain Text Input")
    idea = st.text_area("Research Idea", placeholder="e.g., Improve transformer efficiency for long context language modeling", height=100)
    if idea:
        research_input = idea
else:
    st.header("Structured Input")
    col1, col2 = st.columns(2)
    with col1:
        domain = st.text_input("Domain", placeholder="e.g., NLP")
        task = st.text_input("Task", placeholder="e.g., Text Generation")
    with col2:
        constraint = st.text_input("Constraint", placeholder="e.g., Low compute budget (single GPU)")
    
    if domain or task or constraint:
        research_input = {
            "domain": domain,
            "task": task,
            "constraint": constraint
        }

if st.button("Generate Plan", type="primary"):
    if not research_input:
        st.warning("Please provide a research idea or structured input.")
    else:
        with st.spinner("Agents are analyzing literature and designing the experiment plan... This may take a minute."):
            try:
                state = run_pipeline(research_input)
                output = format_final_output(state)
                
                st.success("Experiment Plan Generated Successfully!")
                
                st.subheader("📋 Problem Statement")
                st.info(output.get('problem_statement', 'N/A'))
                
                # Scores
                st.subheader("📊 Evaluation Scores")
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("Novelty", f"{output.get('novelty_score', 0):.1f}/10")
                sc2.metric("Feasibility", f"{output.get('feasibility_score', 0):.1f}/10")
                sc3.metric("Impact", f"{output.get('impact_score', 0):.1f}/10")
                sc4.metric("Composite", f"{output.get('composite_score', 0):.2f}/10")
                
                verdict = output.get('overall_recommendation', 'N/A').upper()
                sc5.metric("Verdict", verdict)
                
                # Literature Summary
                with st.expander("📚 Literature Summary", expanded=False):
                    st.write(output.get('literature_summary', 'N/A'))
                    
                # Identified Gaps
                st.subheader("🔍 Identified Gaps")
                for i, gap in enumerate(output.get("identified_gaps", []), 1):
                    st.markdown(f"**{i}.** {gap}")
                    
                # Hypotheses
                st.subheader("💡 Hypotheses")
                for i, h in enumerate(output.get("hypotheses", []), 1):
                    st.markdown(f"**{i}.** {h}")
                    
                # Experiment Plan
                st.subheader("🧪 Experiment Plan")
                plan = output.get("experiment_plan", {})
                st.markdown(f"**Architecture:** {plan.get('model_architecture', 'N/A')}")
                st.markdown(f"**Training Strategy:** {plan.get('training_strategy', 'N/A')}")
                
                baselines = plan.get('baseline_models', [])
                baselines_str = ', '.join(baselines) if isinstance(baselines, list) else str(baselines)
                st.markdown(f"**Baselines:** {baselines_str}")
                
                metrics = plan.get('evaluation_metrics', [])
                metrics_str = ', '.join(metrics) if isinstance(metrics, list) else str(metrics)
                st.markdown(f"**Evaluation Metrics:** {metrics_str}")
                
                # Datasets
                st.subheader("📦 Datasets")
                for d in output.get("datasets", []):
                    st.markdown(f"- {d}")
                    
                # Expected Outcomes
                st.subheader("🎯 Expected Outcomes")
                st.success(output.get('expected_outcomes', 'N/A'))
                
                # Critique
                with st.expander("🔬 Critic's Review", expanded=False):
                    st.write(output.get('critique_summary', 'N/A'))
                    
                # Trace / Meta
                st.divider()
                trace_str = ' → '.join(output.get('agent_trace', [])) if isinstance(output.get('agent_trace'), list) else str(output.get('agent_trace'))
                st.caption(f"🔗 Pipeline Trace: {trace_str}")
                st.caption(f"📄 Papers Retrieved: {output.get('papers_retrieved', 0)}")
                if output.get("trace_id"):
                    st.caption(f"📡 Langfuse Trace ID: {output.get('trace_id')}")
                    
                # Download JSON
                st.download_button(
                    label="Download Full JSON Output",
                    data=json.dumps(output, indent=2, ensure_ascii=False),
                    file_name="experiment_plan.json",
                    mime="application/json"
                )
                    
            except Exception as e:
                import traceback
                st.error(f"Pipeline failed: {str(e)}")
                with st.expander("Exception Traceback"):
                    st.code(traceback.format_exc(), language="python")
