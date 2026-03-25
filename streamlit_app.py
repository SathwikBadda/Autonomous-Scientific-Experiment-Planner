import streamlit as st
import json
import os
from pathlib import Path
import sys
import time

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
    page_title="SciPlanner AI", 
    page_icon="🔬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium, dark, scientific look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: radial-gradient(circle at top right, #1e1e2f, #0f0f12);
        color: #e0e0e0;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4a90e2 !important;
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #357abd !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.4);
    }
    
    .card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }
    
    .hero {
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, rgba(74, 144, 226, 0.1), rgba(144, 19, 254, 0.1));
        border-radius: 20px;
        margin-bottom: 2rem;
    }
    
    .status-container {
        border-left: 4px solid #4a90e2;
        padding: 0.5rem 1rem;
        background: rgba(74, 144, 226, 0.05);
        margin: 0.5rem 0;
    }
    
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# sidebar for configuration/meta
with st.sidebar:
    st.image("https://img.icons8.com/parakeet/128/microscope.png", width=64)
    st.title("SciPlanner v1.1")
    st.markdown("---")
    st.info("The multi-agent system that thinks like a scientist.")
    
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

# Hero Section
st.markdown("""
<div class="hero">
    <h1>🔬 Autonomous Scientific Experiment Planner</h1>
    <p style="font-size: 1.2rem; opacity: 0.8;">Analyze literature, identify research gaps, and design breakthrough experiment plans with elite AI agents.</p>
</div>
""", unsafe_allow_html=True)

# Main input area with Tabs
tab1, tab2, tab3 = st.tabs(["💡 Idea Spark", "🧩 Structured", "📄 Paper Analysis"])

research_input = None
pdf_file_path = None

with tab1:
    st.subheader("What's your research direction?")
    idea = st.text_area(
        "Enter your core idea", 
        placeholder="e.g., Improve transformer efficiency for long context language modeling...", 
        height=150,
        help="Provide a brief description of what you want to investigate."
    )
    if idea:
        research_input = idea

with tab2:
    st.subheader("Detailed Research Parameters")
    c1, c2 = st.columns(2)
    with c1:
        domain = st.text_input("Research Domain", placeholder="e.g., Quantum Computing")
        task = st.text_input("Specific Task", placeholder="e.g., Error Correction")
    with c2:
        constraint = st.text_input("Key Constraint", placeholder="e.g., Low qubit count")
    
    if domain or task or constraint:
        research_input = {
            "domain": domain,
            "task": task,
            "constraint": constraint
        }

with tab3:
    st.subheader("Analyze an Existing Paper")
    st.markdown("Upload a research paper (PDF) to find gaps or propose improvements grounded in the specific text.")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        # Save to temp file
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        pdf_file_path = str(temp_dir / uploaded_file.name)
        with open(pdf_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Context loaded from: **{uploaded_file.name}**")
        
        # In tab 3, if no idea is provided, use the filename or a placeholder
        if not research_input:
            st.info("Don't forget to enter a research goal or 'Improve this paper' in the Idea Spark tab, or I'll analyze the paper's core topic.")
            research_input = f"Analyze and improve: {uploaded_file.name}"

st.divider()

if st.button("🚀 Run Scientific Pipeline", type="primary"):
    if not research_input:
        st.error("Please provide a research idea, structured input, or upload a paper to begin.")
    else:
        results_container = st.container()
        
        with st.status("🔬 AI Agents Orchestrating...", expanded=True) as status:
            st.write("📖 Reading global research trends...")
            time.sleep(1)
            
            try:
                state = run_pipeline(research_input, pdf_file_path=pdf_file_path)
                output = format_final_output(state)
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                
                # --- RESULTS DISPLAY ---
                with results_container:
                    st.success("Generation Complete. Review your experiment protocol below.")
                    
                    # Problem & Scores
                    col_left, col_right = st.columns([2, 1])
                    
                    with col_left:
                        st.markdown(f"### 📋 Problem Statement")
                        st.markdown(f'<div class="card">{output.get("problem_statement", "N/A")}</div>', unsafe_allow_html=True)
                    
                    with col_right:
                        st.markdown("### 📊 Overall Scores")
                        sc1, sc2 = st.columns(2)
                        sc1.metric("Novelty", f"{output.get('novelty_score', 0):.1f}")
                        sc2.metric("Impact", f"{output.get('impact_score', 0):.1f}")
                        
                        sc3, sc4 = st.columns(2)
                        sc3.metric("Feasibility", f"{output.get('feasibility_score', 0):.1f}")
                        sc4.metric("Composite", f"{output.get('composite_score', 0):.2f}")
                        
                        st.markdown(f"**Verdict:** :blue[{output.get('overall_recommendation', 'N/A').upper()}]")

                    # Gaps & Hypotheses
                    st.divider()
                    g1, h1 = st.columns(2)
                    
                    with g1:
                        st.subheader("🔍 Key Gaps")
                        for i, gap in enumerate(output.get("identified_gaps", []), 1):
                            with st.expander(f"Gap {i}: {gap[:40]}..."):
                                st.write(gap)
                    
                    with h1:
                        st.subheader("💡 Hypotheses")
                        for i, hyp in enumerate(output.get("hypotheses", []), 1):
                            with st.expander(f"Hypothesis {i}"):
                                st.info(hyp)

                    # Experiment Plan (Main highlight)
                    st.subheader("🧪 Experiment Protocol")
                    plan = output.get("experiment_plan", {})
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="card">
                            <p><strong>Architecture:</strong> {plan.get('model_architecture', 'N/A')}</p>
                            <p><strong>Training Strategy:</strong> {plan.get('training_strategy', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        pc1, pc2 = st.columns(2)
                        with pc1:
                            st.write("**Baselines:**")
                            for b in plan.get('baseline_models', []):
                                st.markdown(f"- {b}")
                        with pc2:
                            st.write("**Target Metrics:**")
                            for m in plan.get('evaluation_metrics', []):
                                st.markdown(f"- {m}")

                    # Literature & Datasets
                    st.divider()
                    l1, d1 = st.columns(2)
                    
                    with l1:
                        st.subheader("📚 Literature Context")
                        st.caption(output.get('literature_summary', 'N/A'))
                    
                    with d1:
                        st.subheader("📦 Recommended Datasets")
                        for d in output.get("datasets", []):
                            st.markdown(f"• {d}")

                    # Outcomes & Critique
                    with st.expander("🎯 Expected Outcomes & Peer Review Critique"):
                        st.success(f"**Outcomes:** {output.get('expected_outcomes', 'N/A')}")
                        st.warning(f"**Critic says:** {output.get('critique_summary', 'N/A')}")

                    # Meta & Trace
                    st.divider()
                    st.caption(f"🔗 Pipeline Trace: {' → '.join(output.get('agent_trace', []))}")
                    
                    st.download_button(
                        label="📥 Download Full Research Report (JSON)",
                        data=json.dumps(output, indent=2),
                        file_name="research_plan.json",
                        mime="application/json"
                    )

            except Exception as e:
                import traceback
                st.error(f"Pipeline crashed during execution.")
                st.exception(e)
                with st.expander("Detailed Traceback"):
                    st.code(traceback.format_exc())
