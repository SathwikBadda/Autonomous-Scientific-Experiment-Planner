"""
tests/test_pipeline.py - Unit and integration tests.
Run with: pytest tests/ -v
"""
import json
import pytest
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────
# Unit: Config loading
# ─────────────────────────────────────────────
def test_config_loads():
    from config.settings import config
    assert "llm" in config
    assert "arxiv" in config
    assert "rag" in config
    assert "agents" in config


def test_prompts_load():
    from config.settings import prompts
    required_agents = [
        "planner_agent", "retrieval_agent", "paper_analyzer_agent",
        "gap_detection_agent", "hypothesis_generator_agent",
        "experiment_planner_agent", "dataset_recommender_agent", "critic_agent",
    ]
    for agent_key in required_agents:
        assert agent_key in prompts, f"Missing prompt for: {agent_key}"
        assert "system" in prompts[agent_key]
        assert "user" in prompts[agent_key]


# ─────────────────────────────────────────────
# Unit: Tool registry
# ─────────────────────────────────────────────
def test_tool_registry():
    from tools.arxiv_tool import TOOL_REGISTRY, get_all_tool_schemas
    assert "search_arxiv" in TOOL_REGISTRY
    schemas = get_all_tool_schemas()
    assert len(schemas) >= 1
    assert schemas[0]["name"] == "search_arxiv"


def test_tool_dispatch_invalid():
    from tools.arxiv_tool import dispatch_tool_call
    with pytest.raises(ValueError, match="Unknown tool"):
        dispatch_tool_call("nonexistent_tool", {})


# ─────────────────────────────────────────────
# Unit: Output formatter
# ─────────────────────────────────────────────
def test_output_formatter_minimal():
    from utils.output_formatter import format_final_output
    state = {
        "problem_statement": "Test problem",
        "literature_summary": "Test summary",
        "identified_gaps": [{"gap": "Gap 1", "gap_type": "experimental"}],
        "hypotheses": [{"hypothesis": "IF A THEN B BECAUSE C"}],
        "experiment_plan": {
            "model_architecture": "Transformer",
            "training_strategy": {"optimizer": "Adam"},
            "baseline_models": [{"name": "BERT"}],
            "evaluation_metrics": ["F1", "BLEU"],
        },
        "datasets": [{"name": "SQuAD", "url": "https://example.com"}],
        "expected_outcomes": "Improved performance",
        "novelty_score": 7,
        "feasibility_score": 8,
        "impact_score": 6,
        "composite_score": 7.15,
        "overall_recommendation": "accept",
        "critique": {"strengths": ["Novel"], "weaknesses": []},
        "agent_trace": ["PlannerAgent", "CriticAgent"],
        "total_papers_fetched": 5,
    }
    output = format_final_output(state)
    assert output["problem_statement"] == "Test problem"
    assert isinstance(output["identified_gaps"], list)
    assert isinstance(output["hypotheses"], list)
    assert 0 <= output["novelty_score"] <= 10
    assert 0 <= output["feasibility_score"] <= 10
    assert 0 <= output["impact_score"] <= 10


def test_output_formatter_score_clamp():
    from utils.output_formatter import format_final_output
    state = {
        "novelty_score": 15,      # should clamp to 10
        "feasibility_score": -3,  # should clamp to 0
        "impact_score": "bad",    # should default to 0
    }
    output = format_final_output(state)
    assert output["novelty_score"] == 10.0
    assert output["feasibility_score"] == 0.0
    assert output["impact_score"] == 0.0


# ─────────────────────────────────────────────
# Unit: RAG pipeline (mocked embeddings)
# ─────────────────────────────────────────────
def test_rag_index_and_search():
    """Test FAISS index build and similarity search with mock embeddings."""
    import numpy as np
    from unittest.mock import patch, MagicMock
    import rag.pipeline as rp

    # Reset state
    rp._faiss_index = None
    rp._stored_papers = []
    rp._embedding_model = None

    # Mock embedding model
    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(3, 1024).astype("float32")

    with patch("rag.pipeline._load_embedding_model", return_value=mock_model):
        with patch("rag.pipeline.config", {
            "rag": {
                "embedding_dim": 1024,
                "faiss_index_type": "FlatIP",
                "top_k": 2,
                "embedding_model_path": "/mock/path",
            }
        }):
            papers = [
                {"title": "Paper A", "abstract": "Abstract A", "arxiv_id": "2401.00001"},
                {"title": "Paper B", "abstract": "Abstract B", "arxiv_id": "2401.00002"},
                {"title": "Paper C", "abstract": "Abstract C", "arxiv_id": "2401.00003"},
            ]
            n = rp.index_papers(papers)
            assert n == 3

            mock_model.encode.return_value = np.random.rand(1, 1024).astype("float32")
            results = rp.search_similar_papers("transformer efficiency", top_k=2)
            assert len(results) <= 2
            for r in results:
                assert "title" in r
                assert "similarity_score" in r


# ─────────────────────────────────────────────
# Unit: Planner Agent (mocked LLM)
# ─────────────────────────────────────────────
def test_planner_agent_run():
    from agents.planner_agent import PlannerAgent
    agent = PlannerAgent()

    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(
            type="text",
            text=json.dumps({
                "problem_statement": "Test problem",
                "domain": "NLP",
                "sub_domain": "Text Generation",
                "task": "Language Modeling",
                "constraints": ["Low compute"],
                "open_challenges": ["Long context"],
                "search_queries": ["efficient transformers", "long context NLP"],
            })
        )
    ]

    with patch.object(agent, "_call_llm", return_value=mock_response):
        state = agent.run(
            {"research_input": "Improve transformer efficiency for long context"},
            trace=MagicMock(),
        )

    assert state["problem_statement"] == "Test problem"
    assert state["domain"] == "NLP"
    assert "efficient transformers" in state["search_queries"]
    assert "PlannerAgent" in state["agent_trace"]


# ─────────────────────────────────────────────
# Unit: Critic Agent scoring
# ─────────────────────────────────────────────
def test_critic_agent_scores():
    from agents.critic_agent import CriticAgent
    agent = CriticAgent()

    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(
            type="text",
            text=json.dumps({
                "critique": {
                    "strengths": ["Novel approach"],
                    "weaknesses": [{"issue": "Limited baselines", "severity": "minor", "suggested_fix": "Add more baselines"}],
                    "revised_suggestions": {},
                },
                "scores": {
                    "novelty_score": 7,
                    "novelty_reasoning": "Addresses known gap",
                    "feasibility_score": 8,
                    "feasibility_reasoning": "Achievable in 12 weeks",
                    "impact_score": 6,
                    "impact_reasoning": "Moderate impact on field",
                },
                "overall_recommendation": "accept",
                "expected_outcomes": "5-10% improvement over baseline",
            })
        )
    ]

    with patch.object(agent, "_call_llm", return_value=mock_response):
        state = agent.run(
            {
                "problem_statement": "Test",
                "literature_summary": "Summary",
                "identified_gaps": [],
                "hypotheses": [],
                "experiment_plan": {},
                "datasets": [],
                "constraints": [],
            },
            trace=MagicMock(),
        )

    assert state["novelty_score"] == 7
    assert state["feasibility_score"] == 8
    assert state["impact_score"] == 6
    assert state["overall_recommendation"] == "accept"
    assert state["composite_score"] > 0


# ─────────────────────────────────────────────
# Integration: FastAPI endpoints
# ─────────────────────────────────────────────
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from api.app import app
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "service" in resp.json()


def test_plan_endpoint_missing_input(client):
    resp = client.post("/plan", json={})
    assert resp.status_code == 422


@patch("api.app.run_pipeline")
@patch("api.app.format_final_output")
def test_plan_endpoint_plain(mock_format, mock_pipeline, client):
    mock_pipeline.return_value = {}
    mock_format.return_value = {
        "problem_statement": "Test",
        "literature_summary": "Summary",
        "identified_gaps": ["Gap 1"],
        "hypotheses": ["Hypothesis 1"],
        "experiment_plan": {
            "model_architecture": "Transformer",
            "training_strategy": "Adam optimizer",
            "baseline_models": ["BERT"],
            "evaluation_metrics": ["F1"],
        },
        "datasets": ["SQuAD"],
        "expected_outcomes": "Better performance",
        "novelty_score": 7.0,
        "feasibility_score": 8.0,
        "impact_score": 6.0,
        "composite_score": 7.15,
        "overall_recommendation": "accept",
        "critique_summary": "Strong approach",
        "agent_trace": ["PlannerAgent", "CriticAgent"],
        "papers_retrieved": 5,
        "trace_id": None,
        "error": None,
    }

    resp = client.post("/plan", json={"plain_input": "Test research idea"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["problem_statement"] == "Test"
    assert data["novelty_score"] == 7.0
