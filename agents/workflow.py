"""
agents/workflow.py - LangGraph multi-agent pipeline.
Defines the full DAG with typed state and agent-to-agent communication.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.critic_agent import CriticAgent
from agents.dataset_recommender_agent import DatasetRecommenderAgent
from agents.experiment_planner_agent import ExperimentPlannerAgent
from agents.gap_detection_agent import GapDetectionAgent
from agents.hypothesis_generator_agent import HypothesisGeneratorAgent
from agents.paper_analyzer_agent import PaperAnalyzerAgent
from agents.planner_agent import PlannerAgent
from agents.retrieval_agent import RetrievalAgent
from rag import session_manager as sm
from utils.logger import get_logger
from utils.tracer import agent_span, create_trace, flush_traces

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Shared Pipeline State (agent-to-agent comms)
# ─────────────────────────────────────────────
class PipelineState(TypedDict, total=False):
    # Input
    research_input: Any

    # Planner output
    problem_statement: str
    domain: str
    sub_domain: str
    task: str
    constraints: list[str]
    open_challenges: list[str]
    search_queries: list[str]
    research_scope: dict

    # Retrieval output
    raw_papers: list[dict]
    retrieved_papers: list[dict]
    total_papers_fetched: int

    # Paper Analyzer output
    paper_analyses: list[dict]
    common_methods: list[str]
    common_datasets: list[str]
    performance_frontier: str
    literature_summary: str

    # Gap Detection output
    identified_gaps: list[dict]
    priority_gaps: list[str]
    gap_summary: str

    # Hypothesis Generator output
    hypotheses: list[dict]
    primary_hypothesis: str
    hypothesis_rationale: str

    # Experiment Planner output
    experiment_plan: dict

    # Dataset Recommender output
    datasets: list[dict]
    data_collection_needed: bool
    data_collection_plan: str

    # Critic output
    critique: dict
    novelty_score: float
    feasibility_score: float
    impact_score: float
    composite_score: float
    overall_recommendation: str
    expected_outcomes: str

    # PDF context
    pdf_context: str
    pdf_chunks: list

    # Session
    session: dict
    session_dir: Optional[str]
    evaluation_metrics_found: list
    proposed_improvements: list

    # Meta
    agent_trace: list
    error: Optional[str]
    trace_id: Optional[str]


# ─────────────────────────────────────────────
# Instantiate agents (singletons)
# ─────────────────────────────────────────────
_planner = PlannerAgent()
_retrieval = RetrievalAgent()
_analyzer = PaperAnalyzerAgent()
_gap = GapDetectionAgent()
_hypothesis = HypothesisGeneratorAgent()
_exp_planner = ExperimentPlannerAgent()
_dataset = DatasetRecommenderAgent()
_critic = CriticAgent()


# ─────────────────────────────────────────────
# Node wrapper factory (injects trace into each agent)
# ─────────────────────────────────────────────
def _make_node(agent, trace_holder: dict):
    def node_fn(state: PipelineState) -> PipelineState:
        trace = trace_holder.get("trace")
        try:
            return agent.run(state, trace=trace)
        except Exception as exc:
            logger.error(
                "node_error",
                agent=agent.agent_name,
                error=str(exc),
            )
            return {**state, "error": str(exc)}

    node_fn.__name__ = agent.agent_name
    return node_fn


# ─────────────────────────────────────────────
# Build the LangGraph StateGraph
# ─────────────────────────────────────────────
def build_graph(trace_holder: dict) -> StateGraph:
    """
    Constructs the LangGraph DAG.
    trace_holder is a mutable dict so the Langfuse trace can be
    injected after graph construction (before invoke).
    """
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("planner",             _make_node(_planner,     trace_holder))
    graph.add_node("retrieval",           _make_node(_retrieval,   trace_holder))
    graph.add_node("paper_analyzer",      _make_node(_analyzer,    trace_holder))
    graph.add_node("gap_detection",       _make_node(_gap,         trace_holder))
    graph.add_node("hypothesis_generator",_make_node(_hypothesis,  trace_holder))
    graph.add_node("experiment_planner",  _make_node(_exp_planner, trace_holder))
    graph.add_node("dataset_recommender", _make_node(_dataset,     trace_holder))
    graph.add_node("critic",              _make_node(_critic,      trace_holder))

    # Linear DAG edges
    graph.add_edge(START,                  "planner")
    graph.add_edge("planner",              "retrieval")
    graph.add_edge("retrieval",            "paper_analyzer")
    graph.add_edge("paper_analyzer",       "gap_detection")
    graph.add_edge("gap_detection",        "hypothesis_generator")
    graph.add_edge("hypothesis_generator", "experiment_planner")
    graph.add_edge("experiment_planner",   "dataset_recommender")
    graph.add_edge("dataset_recommender",  "critic")
    graph.add_edge("critic",               END)

    return graph.compile()


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────
def run_pipeline(research_input: Any, pdf_file_path: Optional[str] = None) -> dict:
    """
    Execute the full multi-agent pipeline.
    Creates a session folder, resets FAISS, preprocesses PDF if provided,
    runs the agent graph, saves the FAISS index to disk, and returns the final state.
    """
    from rag.pipeline import reset_index, index_chunks, search_pdf_chunks, build_rag_context, save_index_to_disk
    from rag.document_loader import PDFLoader

    logger.info("pipeline_start", input=str(research_input)[:200], pdf=pdf_file_path)

    # 1. Create session folder
    problem_statement_hint = (
        research_input if isinstance(research_input, str)
        else research_input.get("task", research_input.get("domain", ""))
    )
    session = sm.create_session(problem_statement=problem_statement_hint)
    logger.info("session_ready", session_id=session["session_id"])

    # 2. Reset FAISS for a fresh run
    reset_index()

    pdf_chunks = []
    pdf_context = ""

    if pdf_file_path:
        try:
            raw_text = PDFLoader.extract_text(pdf_file_path)
            pdf_chunks = PDFLoader.chunk_text(raw_text)
            index_chunks(pdf_chunks)
            query = research_input if isinstance(research_input, str) else research_input.get("task", "")
            relevant_chunks = search_pdf_chunks(query, top_k=5)
            pdf_context = build_rag_context(chunks=relevant_chunks)
        except Exception as e:
            logger.error("pdf_preprocessing_failed", error=str(e))

    # 3. Create Langfuse trace
    trace = create_trace(
        name="sci_planner_pipeline",
        metadata={
            "input": str(research_input)[:500],
            "session_id": session["session_id"],
            "has_pdf": bool(pdf_file_path),
        },
    )
    trace_holder = {"trace": trace}

    # 4. Build and run graph
    app = build_graph(trace_holder)

    initial_state: PipelineState = {
        "research_input": research_input,
        "pdf_chunks": pdf_chunks,
        "pdf_context": pdf_context,
        "session": session,
        "session_dir": session["session_dir"],
        "agent_trace": [],
        "error": None,
        "trace_id": getattr(trace, "id", None),
    }

    final_state = app.invoke(initial_state)

    # 5. Save FAISS index to disk
    try:
        saved = save_index_to_disk(session["index_dir"])
        logger.info("faiss_index_saved", files=saved, session=session["session_id"])
    except Exception as e:
        logger.warning("faiss_index_save_failed", error=str(e))

    # 6. Update Langfuse trace with scores
    try:
        trace.update(
            output={
                "novelty_score": final_state.get("novelty_score"),
                "feasibility_score": final_state.get("feasibility_score"),
                "impact_score": final_state.get("impact_score"),
                "composite_score": final_state.get("composite_score"),
                "recommendation": final_state.get("overall_recommendation"),
                "session_id": session["session_id"],
                "papers_retrieved": final_state.get("total_papers_fetched", 0),
            }
        )
    except Exception:
        pass

    flush_traces()

    logger.info(
        "pipeline_complete",
        agents_run=final_state.get("agent_trace", []),
        novelty=final_state.get("novelty_score"),
        feasibility=final_state.get("feasibility_score"),
        impact=final_state.get("impact_score"),
        session_id=session["session_id"],
    )
    return dict(final_state)
