import streamlit as st
import json
import os
from pathlib import Path
import sys
import time

project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agents.workflow import run_pipeline
from utils.output_formatter import format_final_output
from config.settings import config
from utils.logger import setup_logging

setup_logging()

st.set_page_config(
    page_title="SciPlanner AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: radial-gradient(circle at top right, #1a1a2e, #0d0d15); color: #e0e0e0; }

.stButton>button {
    width: 100%; border-radius: 8px; height: 3.2em;
    background: linear-gradient(135deg, #4a90e2, #7b2ff7) !important;
    color: white; font-weight: 600; border: none; transition: all 0.3s ease;
    font-size: 1.05rem;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(74, 144, 226, 0.5);
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 1.2rem 1.5rem; border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.1); margin-bottom: 1rem;
}

.section-header {
    font-size: 1.1rem; font-weight: 700; color: #ffffff;
    margin: 1.2rem 0 0.6rem 0; padding-bottom: 0.3rem;
    border-bottom: 2px solid rgba(74,144,226,0.4);
}

.bullet-item {
    padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 0.95rem; line-height: 1.6;
}

.dataset-card {
    background: rgba(74,144,226,0.08); border: 1px solid rgba(74,144,226,0.3);
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}

.model-card {
    background: rgba(123,47,247,0.08); border: 1px solid rgba(123,47,247,0.3);
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}

.score-badge {
    display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px;
    font-weight: 700; font-size: 1.1rem;
}

.gap-badge {
    display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px;
    font-size: 0.75rem; font-weight: 600; margin-left: 0.4rem;
}

.hero {
    text-align: center; padding: 2.5rem 1rem;
    background: linear-gradient(135deg, rgba(74,144,226,0.12), rgba(123,47,247,0.12));
    border-radius: 20px; margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.08);
}

h1, h2, h3 { color: #ffffff !important; font-weight: 700; }

.step-item {
    display: flex; align-items: flex-start; padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.resource-row {
    display: flex; justify-content: space-between;
    padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.06);
}

.stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/parakeet/128/microscope.png", width=60)
    st.title("SciPlanner v2.0")
    st.markdown("---")
    st.info("Multi-agent system that thinks like a scientist.\n\nGives your idea a complete research shape.")
    st.markdown("---")
    st.caption("**Pipeline:**\nPlanner → Retrieval → Paper Analyzer → Gap Detection → Hypothesis → Experiment Planner → Datasets → Critic")
    if st.button("🔄 Reset Session"):
        st.session_state.clear()
        st.rerun()

# Hero
st.markdown("""
<div class="hero">
    <h1>🔬 Autonomous Scientific Experiment Planner</h1>
    <p style="font-size:1.15rem;opacity:0.85;max-width:700px;margin:0.8rem auto 0;">
        Give me any research idea. I'll analyze the literature, identify gaps, generate hypotheses,
        design the full experiment, and give your idea a complete research shape.
    </p>
</div>
""", unsafe_allow_html=True)

# Input tabs
tab1, tab2, tab3 = st.tabs(["💡 Research Idea", "🧩 Structured Input", "📄 Paper Upload"])

research_input = None
pdf_file_path = None

with tab1:
    st.subheader("What research problem do you want to solve?")
    idea = st.text_area(
        "Enter your research idea",
        placeholder="e.g., Improve object detection accuracy in low-light conditions for autonomous driving...",
        height=130,
        help="Be as specific as possible. The agents will expand and plan around your idea."
    )
    if idea:
        research_input = idea

with tab2:
    st.subheader("Detailed Research Parameters")
    c1, c2 = st.columns(2)
    with c1:
        domain = st.text_input("Research Domain", placeholder="e.g., Computer Vision")
        task = st.text_input("Specific Task", placeholder="e.g., Object Detection in Low Light")
    with c2:
        constraint = st.text_input("Key Constraint", placeholder="e.g., Real-time inference on edge devices")
    if domain or task:
        research_input = {"domain": domain, "task": task, "constraint": constraint}

with tab3:
    st.subheader("Upload a Research Paper")
    uploaded_file = st.file_uploader("Upload PDF to ground the analysis in a specific paper", type=["pdf"])
    if uploaded_file:
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        pdf_file_path = str(temp_dir / uploaded_file.name)
        with open(pdf_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Loaded: **{uploaded_file.name}**")
        if not research_input:
            research_input = f"Analyze and build on: {uploaded_file.name}"

st.divider()

if st.button("🚀 Generate Research Plan", type="primary"):
    if not research_input:
        st.error("Please enter a research idea or upload a paper.")
    else:
        with st.status("🔬 AI Agents Working...", expanded=True) as status:
            st.write("🧠 Planner scoping the problem...")
            st.write("🔍 Retrieval agent fetching literature...")
            st.write("📊 Analyzing papers, detecting gaps...")
            st.write("💡 Generating hypotheses and experiment plan...")
            try:
                state = run_pipeline(research_input, pdf_file_path=pdf_file_path)
                output = format_final_output(state)
                status.update(label="✅ Research Plan Complete!", state="complete", expanded=False)
            except Exception as e:
                import traceback
                status.update(label="❌ Pipeline Error", state="error")
                st.error("Pipeline crashed.")
                st.exception(e)
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
                st.stop()

        st.success("✅ Generation Complete. Review your research plan below.")

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 1: PROBLEM STATEMENT + SCORES
        # ═══════════════════════════════════════════════════════════════════
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown("### 🧠 Problem Statement")
            st.markdown(f'<div class="card">{output.get("problem_statement", "N/A")}</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown("### 📊 Research Quality Scores")
            n, f, i, c = (
                output.get("novelty_score", 0),
                output.get("feasibility_score", 0),
                output.get("impact_score", 0),
                output.get("composite_score", 0),
            )
            sc1, sc2 = st.columns(2)
            sc1.metric("Novelty", f"{n:.1f}/10")
            sc2.metric("Impact", f"{i:.1f}/10")
            sc3, sc4 = st.columns(2)
            sc3.metric("Feasibility", f"{f:.1f}/10")
            sc4.metric("Composite", f"{c:.2f}")
            rec = output.get("overall_recommendation", "revise").upper()
            color = "green" if rec == "ACCEPT" else ("red" if rec == "REJECT" else "blue")
            st.markdown(f"**Verdict:** :{color}[{rec}]")

        st.divider()

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 2: LITERATURE SUMMARY
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("### 📚 Literature Summary")
        lit_summary = output.get("literature_summary", "")
        lit_bullets = output.get("literature_summary_bullets", [])

        col_lit, col_papers = st.columns([3, 2])
        with col_lit:
            if lit_bullets:
                for bullet in lit_bullets:
                    st.markdown(f'<div class="bullet-item">• {bullet}</div>', unsafe_allow_html=True)
            elif lit_summary and lit_summary not in ("No papers retrieved.", ""):
                for line in lit_summary.split("\n"):
                    if line.strip():
                        st.markdown(f'<div class="bullet-item">• {line.strip()}</div>', unsafe_allow_html=True)
            else:
                st.caption("Literature summary will appear after paper retrieval.")

        with col_papers:
            top_papers = output.get("top_papers", [])
            if top_papers:
                st.markdown("**🔗 Top Retrieved Papers**")
                for p in top_papers[:3]:
                    url = p.get("url", "")
                    title = p.get("title", "Unknown")
                    year = p.get("year", "")
                    if url:
                        st.markdown(f"• [{title[:55]}...]({url}) ({year})")
                    else:
                        st.markdown(f"• {title[:60]}... ({year})")

        st.divider()

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 3: GAPS + HYPOTHESES (matching reference layout)
        # ═══════════════════════════════════════════════════════════════════
        g_col, h_col = st.columns(2)

        with g_col:
            st.markdown("### 🔍 Identified Gaps")
            gaps_rich = output.get("identified_gaps_rich", [])
            gaps_simple = output.get("identified_gaps", [])

            if gaps_rich:
                for i, g in enumerate(gaps_rich[:6], 1):
                    gap_text = g.get("gap", str(g)) if isinstance(g, dict) else str(g)
                    gap_type = g.get("gap_type", "") if isinstance(g, dict) else ""
                    difficulty = g.get("difficulty", "") if isinstance(g, dict) else ""
                    diff_color = {"easy": "#27ae60", "medium": "#f39c12", "hard": "#e74c3c"}.get(difficulty, "#888")
                    with st.expander(f"Gap {i}: {gap_text[:55]}...", expanded=(i <= 2)):
                        st.write(gap_text)
                        if g.get("evidence"): st.caption(f"📄 Evidence: {g['evidence']}")
                        if g.get("proposed_direction"): st.info(f"→ {g['proposed_direction']}")
                        cols = st.columns(2)
                        if gap_type: cols[0].caption(f"Type: `{gap_type}`")
                        if difficulty: cols[1].caption(f"Difficulty: `{difficulty}`")
            elif gaps_simple:
                for i, g in enumerate(gaps_simple[:6], 1):
                    st.markdown(f'<div class="bullet-item">• {g}</div>', unsafe_allow_html=True)
            else:
                st.info("No gaps detected. Check paper retrieval logs.")

        with h_col:
            st.markdown("### 💡 Hypotheses")
            hyps_rich = output.get("hypotheses_rich", [])
            hyps_simple = output.get("hypotheses", [])

            if hyps_rich:
                for i, h in enumerate(hyps_rich[:4], 1):
                    hyp_text = h.get("hypothesis", str(h)) if isinstance(h, dict) else str(h)
                    approach = h.get("approach", "") if isinstance(h, dict) else ""
                    improvement = h.get("expected_improvement", "") if isinstance(h, dict) else ""
                    confidence = h.get("confidence", "") if isinstance(h, dict) else ""
                    with st.expander(f"H{i}: {hyp_text[:50]}...", expanded=(i == 1)):
                        st.info(hyp_text)
                        if approach: st.markdown(f"**Approach:** {approach}")
                        if improvement: st.success(f"📈 Expected: {improvement}")
                        if confidence: st.caption(f"Confidence: `{confidence}`")
            elif hyps_simple:
                for i, h in enumerate(hyps_simple[:4], 1):
                    st.markdown(f'<div class="bullet-item">• {h}</div>', unsafe_allow_html=True)
            else:
                st.info("No hypotheses generated.")

        st.divider()

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 4: EXPERIMENT PLAN
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("### 🧪 Experiment Plan")
        plan = output.get("experiment_plan", {})
        raw_plan_state = state.get("experiment_plan", {})
        if "experiment_plan" in raw_plan_state:
            raw_plan_state = raw_plan_state["experiment_plan"]

        exp1, exp2 = st.columns([3, 2])

        with exp1:
            st.markdown("**Model Architecture**")
            arch_text = plan.get("model_architecture", "Not specified")
            st.markdown(f'<div class="card">{arch_text}</div>', unsafe_allow_html=True)

            # Recommended Models
            rec_models = output.get("recommended_models", []) or raw_plan_state.get("recommended_models", [])
            if rec_models:
                st.markdown("**Recommended Models**")
                for model in rec_models:
                    if isinstance(model, dict):
                        role = model.get("role", "")
                        name = model.get("model_name", "")
                        just = model.get("justification", "")
                        params = model.get("parameters", "")
                        pretrained = model.get("pretrained_on", "")
                        st.markdown(f"""
<div class="model-card">
<strong>🔧 {role}:</strong> <code>{name}</code><br>
<small>{just}</small>
{f'<br><small>📦 Params: {params} | Pre-trained: {pretrained}</small>' if params else ''}
</div>""", unsafe_allow_html=True)

            # Key Components
            components = plan.get("key_components", []) or raw_plan_state.get("key_components", [])
            if components:
                st.markdown("**Key Components**")
                for c in components:
                    st.markdown(f'<div class="bullet-item">✦ {c}</div>', unsafe_allow_html=True)

        with exp2:
            # Training Strategy
            training_dict = plan.get("training_strategy_dict", {}) or raw_plan_state.get("training_strategy", {})
            if isinstance(training_dict, dict) and training_dict:
                st.markdown("**Training Strategy**")
                for k, v in training_dict.items():
                    nice_k = k.replace("_", " ").title()
                    st.markdown(f'<div class="bullet-item"><strong>{nice_k}:</strong> {v}</div>', unsafe_allow_html=True)
            else:
                training_str = plan.get("training_strategy", "")
                if training_str:
                    st.markdown("**Training Strategy**")
                    st.markdown(f'<div class="card">{training_str}</div>', unsafe_allow_html=True)

            # Resource Requirements
            resources = output.get("resource_requirements", {}) or raw_plan_state.get("resource_requirements", {})
            if resources:
                st.markdown("**⚙️ Resource Requirements**")
                for k, v in resources.items():
                    nice_k = k.replace("_", " ").title()
                    st.markdown(f'<div class="bullet-item"><strong>{nice_k}:</strong> {v}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Baselines + Metrics side by side
        bm1, bm2 = st.columns(2)
        with bm1:
            st.markdown("**Baselines**")
            baselines = plan.get("baseline_models", []) or raw_plan_state.get("baseline_models", [])
            for b in baselines:
                if isinstance(b, dict):
                    st.markdown(f'<div class="bullet-item">• <strong>{b.get("name","")}</strong> — {b.get("why_chosen","")}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bullet-item">• {b}</div>', unsafe_allow_html=True)
        with bm2:
            st.markdown("**Evaluation Metrics**")
            for m in plan.get("evaluation_metrics", []):
                st.markdown(f'<div class="bullet-item">• {m}</div>', unsafe_allow_html=True)

        # Experiment Steps
        steps = plan.get("experiment_steps", []) or raw_plan_state.get("experiment_steps", [])
        if steps:
            with st.expander("📋 Step-by-Step Experiment Execution Plan", expanded=True):
                for step in steps:
                    st.markdown(f"✅ {step}")

        # Ablation studies
        ablations = plan.get("ablation_studies", []) or raw_plan_state.get("ablation_studies", [])
        if ablations:
            with st.expander("🔬 Ablation Studies"):
                for ab in ablations:
                    st.markdown(f"• {ab}")

        st.divider()

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 5: ARCHITECTURE DIAGRAM
        # ═══════════════════════════════════════════════════════════════════
        arch_diagram = output.get("architecture_diagram", "") or state.get("architecture_diagram", "")
        if arch_diagram:
            st.markdown("### 🏗️ Architecture Diagram")
            # Extract mermaid block if it's wrapped in ```mermaid ... ```
            import re
            mermaid_match = re.search(r"```mermaid\n(.+?)```", arch_diagram, re.DOTALL)
            if mermaid_match:
                st.markdown(arch_diagram)
            else:
                st.markdown(f"```mermaid\n{arch_diagram}\n```")
            st.divider()

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 6: DATASET SUGGESTIONS
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("### 📊 Dataset Suggestions")
        datasets_rich = output.get("datasets_rich", [])
        datasets_simple = output.get("datasets", [])

        if datasets_rich and any(isinstance(d, dict) for d in datasets_rich):
            ds_cols = st.columns(2)
            for i, d in enumerate(datasets_rich[:6]):
                if isinstance(d, dict):
                    col = ds_cols[i % 2]
                    with col:
                        name = d.get("name", "Unknown")
                        desc = d.get("description", "")
                        url = d.get("url", "")
                        size = d.get("size", "")
                        use = d.get("use_in_experiment", "")
                        justification = d.get("justification", "")
                        dl_cmd = d.get("download_command", "")

                        url_part = f'<br><a href="{url}" target="_blank" style="color:#4a90e2;font-size:0.85rem;">🔗 {url[:50]}...</a>' if url else ""
                        st.markdown(f"""
<div class="dataset-card">
<strong>📦 {name}</strong>{f' <span style="font-size:0.75rem;color:#aaa;">| {use}</span>' if use else ''}<br>
<small>{desc[:150] if desc else ''}</small>{url_part}
{f'<br><small>📐 Size: {size}</small>' if size else ''}
{f'<br><small>✓ {justification[:100]}</small>' if justification else ''}
</div>""", unsafe_allow_html=True)
                        if dl_cmd:
                            st.code(dl_cmd, language="bash")
        elif datasets_simple:
            for d in datasets_simple:
                st.markdown(f'<div class="bullet-item">• {d}</div>', unsafe_allow_html=True)
        else:
            st.info("No dataset recommendations yet.")

        st.divider()

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 7: EXPECTED OUTCOMES
        # ═══════════════════════════════════════════════════════════════════
        expected = output.get("expected_outcomes", "")
        if expected:
            st.markdown("### 🎯 Expected Outcomes")
            for line in expected.split("\n"):
                if line.strip():
                    st.markdown(f'<div class="bullet-item">• {line.strip()}</div>', unsafe_allow_html=True)
            st.divider()

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 8: PROPOSED IMPROVEMENTS
        # ═══════════════════════════════════════════════════════════════════
        proposed = state.get("proposed_improvements", output.get("proposed_improvements", []))
        if proposed:
            with st.expander("🔧 Proposed Methodological Improvements", expanded=True):
                for i, imp in enumerate(proposed, 1):
                    st.markdown(f"**{i}.** {imp}")

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 9: PEER REVIEW CRITIQUE
        # ═══════════════════════════════════════════════════════════════════
        critique = output.get("critique", {})
        critique_summary = output.get("critique_summary", "")
        with st.expander("🎓 Peer Review Critique", expanded=False):
            if critique_summary:
                st.warning(f"**Overall Assessment:** {critique_summary}")

            if critique:
                c_strengths = critique.get("strengths", [])
                c_weaknesses = critique.get("weaknesses", [])

                cr1, cr2 = st.columns(2)
                with cr1:
                    if c_strengths:
                        st.markdown("**✅ Strengths**")
                        for s in c_strengths:
                            st.markdown(f"• {s}")
                with cr2:
                    if c_weaknesses:
                        st.markdown("**⚠️ Weaknesses & Fixes**")
                        for w in c_weaknesses:
                            if isinstance(w, dict):
                                sev = w.get("severity", "")
                                sev_icon = {"critical": "🔴", "major": "🟡", "minor": "🟢"}.get(sev, "⚪")
                                st.markdown(f"{sev_icon} **{w.get('issue', '')}**")
                                if w.get("suggested_fix"):
                                    st.caption(f"→ Fix: {w['suggested_fix']}")
                            else:
                                st.markdown(f"• {w}")

            # Scores breakdown
            st.markdown("---")
            sr1, sr2, sr3 = st.columns(3)
            sr1.metric("Novelty", f"{output.get('novelty_score', 0):.1f}/10")
            sr2.metric("Feasibility", f"{output.get('feasibility_score', 0):.1f}/10")
            sr3.metric("Impact", f"{output.get('impact_score', 0):.1f}/10")

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 10: RESEARCH PAPER OUTLINE
        # ═══════════════════════════════════════════════════════════════════
        paper_outline = output.get("research_paper_outline", "") or state.get("research_paper_outline", "")
        if paper_outline:
            with st.expander("📄 Research Paper Outline", expanded=False):
                st.markdown(paper_outline)

        # ═══════════════════════════════════════════════════════════════════
        # META & DOWNLOAD
        # ═══════════════════════════════════════════════════════════════════
        st.divider()
        session_state_dict = state.get("session", {})
        session_id = session_state_dict.get("session_id", "N/A")
        session_dir = session_state_dict.get("session_dir", "")

        st.caption(f"🔗 Pipeline: {' → '.join(output.get('agent_trace', []))}")
        st.caption(f"📄 Papers Retrieved: {output.get('papers_retrieved', 0)} | Session: `{session_id}`")

        if session_dir:
            with st.expander("📁 Session Files"):
                st.code(session_dir, language="bash")
                st.markdown("All papers saved as **JSON + Markdown** in `papers/`. "
                            "ChromaDB vector index in `index/`.")

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "📥 Download Research Report (JSON)",
                data=json.dumps(output, indent=2, default=str),
                file_name="research_plan.json",
                mime="application/json",
            )
        with dl_col2:
            # Generate markdown report
            md_report = f"""# Research Plan: {output.get('problem_statement', '')[:60]}

## Problem Statement
{output.get('problem_statement', '')}

## Literature Summary
{chr(10).join('- ' + b for b in output.get('literature_summary_bullets', []))}

## Identified Gaps
{chr(10).join('- ' + g for g in output.get('identified_gaps', []))}

## Hypotheses
{chr(10).join('- ' + h for h in output.get('hypotheses', []))}

## Experiment Plan
**Architecture:** {output.get('experiment_plan', {}).get('model_architecture', '')}

**Steps:**
{chr(10).join(output.get('experiment_plan', {}).get('experiment_steps', []))}

## Expected Outcomes
{output.get('expected_outcomes', '')}

## Scores
- Novelty: {output.get('novelty_score', 0):.1f}/10
- Feasibility: {output.get('feasibility_score', 0):.1f}/10
- Impact: {output.get('impact_score', 0):.1f}/10
"""
            st.download_button(
                "📝 Download Report (Markdown)",
                data=md_report,
                file_name="research_plan.md",
                mime="text/markdown",
            )
