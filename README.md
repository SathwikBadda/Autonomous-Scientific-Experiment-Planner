# 🔬 Autonomous Scientific Experiment Planner

> A multi-agent AI system that **thinks like a scientist** — analyzing literature, identifying research gaps, generating hypotheses, and designing rigorous experiment plans.

---

## 📐 Architecture Overview

```
User Input (plain text or JSON)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LangGraph Pipeline                          │
│                                                                 │
│  ┌──────────┐   ┌───────────┐   ┌──────────────┐              │
│  │ Planner  │──▶│ Retrieval │──▶│PaperAnalyzer │              │
│  │  Agent   │   │  Agent    │   │    Agent     │              │
│  │          │   │(MCP Tools)│   │              │              │
│  └──────────┘   └─────┬─────┘   └──────┬───────┘              │
│                        │  arXiv API     │                       │
│                        ▼  + FAISS RAG  ▼                       │
│                  ┌──────────┐   ┌──────────────┐              │
│                  │   Gap    │──▶│  Hypothesis  │              │
│                  │Detection │   │  Generator   │              │
│                  └──────────┘   └──────┬───────┘              │
│                                        │                       │
│  ┌──────────┐   ┌───────────┐   ┌──────▼───────┐              │
│  │  Critic  │◀──│ Dataset   │◀──│  Experiment  │              │
│  │  Agent   │   │Recommender│   │   Planner    │              │
│  └──────────┘   └───────────┘   └──────────────┘              │
│       │                                                         │
└───────┼─────────────────────────────────────────────────────────┘
        │
        ▼
Strict JSON Output (scores + plan)

Langfuse traces every agent span ↑
```

### Agent-to-Agent Communication
All agents share a **typed `PipelineState` dict** managed by LangGraph. Each agent receives the full state from the previous node and appends its results. This enables full traceability of what each agent consumed and produced.

| Agent | Reads From State | Writes To State |
|-------|-----------------|-----------------|
| PlannerAgent | `research_input` | `problem_statement`, `search_queries`, `research_scope` |
| RetrievalAgent | `search_queries`, `research_scope` | `raw_papers`, `retrieved_papers` |
| PaperAnalyzerAgent | `retrieved_papers`, `problem_statement` | `paper_analyses`, `literature_summary`, `common_datasets` |
| GapDetectionAgent | `paper_analyses`, `research_scope` | `identified_gaps`, `priority_gaps` |
| HypothesisGeneratorAgent | `identified_gaps`, `literature_summary` | `hypotheses`, `primary_hypothesis` |
| ExperimentPlannerAgent | `primary_hypothesis`, `hypotheses` | `experiment_plan` |
| DatasetRecommenderAgent | `experiment_plan`, `common_datasets` | `datasets` |
| CriticAgent | full state | `novelty_score`, `feasibility_score`, `impact_score`, `critique` |

---

## 📁 Project Structure

```
autonomous-sci-planner/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py               # BaseAgent class (LLM calls, JSON parsing)
│   ├── planner_agent.py            # Node 1: Parse input → research scope
│   ├── retrieval_agent.py          # Node 2: MCP tool-use arXiv search
│   ├── paper_analyzer_agent.py     # Node 3: Extract methods/gaps from papers
│   ├── gap_detection_agent.py      # Node 4: Identify research gaps
│   ├── hypothesis_generator_agent.py # Node 5: Generate IF/THEN/BECAUSE hypotheses
│   ├── experiment_planner_agent.py # Node 6: Design full experiment protocol
│   ├── dataset_recommender_agent.py # Node 7: Recommend datasets
│   ├── critic_agent.py             # Node 8: Peer-review + scoring
│   └── workflow.py                 # LangGraph StateGraph orchestration
├── api/
│   ├── __init__.py
│   └── app.py                      # FastAPI REST endpoints
├── config/
│   ├── __init__.py
│   ├── config.yaml                 # All tuneable parameters
│   └── settings.py                 # Config loader + env overrides
├── prompts/
│   └── agent_prompts.yaml          # All agent system + user prompts
├── rag/
│   ├── __init__.py
│   └── pipeline.py                 # FAISS index + similarity search
├── tools/
│   ├── __init__.py
│   └── arxiv_tool.py               # MCP tool registry + arXiv executor
├── utils/
│   ├── __init__.py
│   ├── logger.py                   # Structured logging (structlog)
│   ├── output_formatter.py         # State → strict JSON output
│   └── tracer.py                   # Langfuse trace/span management
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py            # Unit + integration tests
├── .env.example                    # Environment variable template
├── example_run.py                  # CLI demo script
├── main.py                         # Entrypoint (serve or CLI run)
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Set Up Environment

```bash
git clone <your-repo-url>
cd autonomous-sci-planner

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download the Embedding Model

The system uses `mixedbread-ai/mxbai-embed-large-v1` loaded from a **local path**.

```bash
# Option A: Using Hugging Face hub download
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
model.save('/your/local/path/mixedbread-ai_mxbai-embed-large-v1')
print('Model saved.')
"

# Option B: Using huggingface-cli
huggingface-cli download mixedbread-ai/mxbai-embed-large-v1 \
  --local-dir /your/local/path/mixedbread-ai_mxbai-embed-large-v1
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
ANTHROPIC_API_KEY=sk-ant-your-key-here
EMBEDDING_MODEL_PATH=/your/local/path/mixedbread-ai_mxbai-embed-large-v1

# Optional: Langfuse observability (recommended)
LANGFUSE_PUBLIC_KEY=pk-lf-your-key
LANGFUSE_SECRET_KEY=sk-lf-your-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 4. Update config.yaml (optional)

Open `config/config.yaml` and adjust:
- `llm.model` — Claude model to use (default: `claude-opus-4-5`)
- `arxiv.max_results` — papers per query (default: 10)
- `rag.top_k` — papers passed to analysis agents (default: 5)
- `langfuse.enabled` — set `true` to enable tracing

### 5. Run the API Server

```bash
python main.py serve
# Server starts at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 6. Run a Quick CLI Example

```bash
python main.py run "Improve transformer efficiency for long context language modeling"

# Or structured input:
python main.py run --domain NLP --task "Text Generation" --constraint "Low compute"

# Or the full example script:
python example_run.py
```

---

## 🌐 API Usage

### POST `/plan` — Main endpoint

```bash
curl -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{"plain_input": "Improve transformer efficiency for long context"}'
```

```bash
# Structured input
curl -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{
    "structured_input": {
      "domain": "NLP",
      "task": "Text Generation",
      "constraint": "Low compute"
    }
  }'
```

### POST `/plan/plain` — Convenience endpoint

```bash
curl -X POST http://localhost:8000/plan/plain \
  -H "Content-Type: application/json" \
  -d '{"research_idea": "Few-shot learning for low-resource NLP"}'
```

### Sample Output

```json
{
  "problem_statement": "Current transformer architectures struggle with...",
  "literature_summary": "Analysis of 8 papers: ...",
  "identified_gaps": [
    "No existing work evaluates sliding window attention beyond 8K tokens on multilingual benchmarks",
    "Missing ablation studies on positional encoding for context > 32K"
  ],
  "hypotheses": [
    "IF we apply hierarchical sparse attention with dynamic routing THEN perplexity improves 12-18% on 64K token sequences BECAUSE ..."
  ],
  "experiment_plan": {
    "model_architecture": "Hierarchical Sparse Transformer with ALiBi positional encoding...",
    "training_strategy": "optimizer: AdamW; lr: 3e-4; batch_size: 32; ...",
    "baseline_models": ["LongFormer", "BigBird", "FlashAttention-2"],
    "evaluation_metrics": ["Perplexity", "ROUGE-L", "Throughput (tokens/sec)"]
  },
  "datasets": ["SCROLLS", "LongBench", "QASPER", "ZeroScrolls"],
  "expected_outcomes": "15-20% efficiency gain with <2% accuracy drop...",
  "novelty_score": 7.5,
  "feasibility_score": 8.0,
  "impact_score": 7.0,
  "composite_score": 7.6,
  "overall_recommendation": "accept"
}
```

---

## 📡 Langfuse Observability

Every pipeline run creates a Langfuse trace with:

- **Top-level trace**: full pipeline run with input/output + scores
- **Per-agent spans**: start time, input, output, duration, errors
- **LLM generations**: prompt, completion, model for every Claude call

### Setup Langfuse

1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com) (free tier available)
2. Create a project → copy Public Key + Secret Key
3. Add to `.env`
4. Set `langfuse.enabled: true` in `config/config.yaml`

### What You'll See

```
Trace: sci_planner_pipeline
  ├─ Span: PlannerAgent          (2.1s)
  ├─ Span: RetrievalAgent        (18.4s)  ← includes arXiv calls
  │    └─ Generation: llm_call   (6.2s)
  ├─ Span: PaperAnalyzerAgent    (8.7s)
  ├─ Span: GapDetectionAgent     (5.3s)
  ├─ Span: HypothesisGenerator   (6.1s)
  ├─ Span: ExperimentPlanner     (7.2s)
  ├─ Span: DatasetRecommender    (4.8s)
  └─ Span: CriticAgent           (9.1s)
```

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=. --cov-report=term-missing

# Single test
pytest tests/test_pipeline.py::test_planner_agent_run -v
```

Test categories:
- **Config loading** — YAML + prompt file integrity
- **Tool registry** — MCP schema validation
- **Output formatter** — score clamping, field normalization
- **RAG pipeline** — FAISS index + search (mocked embeddings)
- **Agent unit tests** — mocked LLM responses
- **FastAPI integration** — endpoint validation

---

## ⚙️ Configuration Reference

### `config/config.yaml`

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `llm` | `model` | `claude-opus-4-5` | Claude model |
| `llm` | `max_tokens` | `4096` | Max output tokens |
| `llm` | `temperature` | `0.3` | Sampling temperature |
| `arxiv` | `max_results` | `10` | Papers per query |
| `rag` | `top_k` | `5` | Top similar papers |
| `rag` | `faiss_index_type` | `FlatIP` | FAISS index type |
| `rag` | `embedding_dim` | `1024` | mxbai-embed-large-v1 dim |
| `langfuse` | `enabled` | `true` | Enable tracing |
| `scoring` | `novelty_weight` | `0.4` | Composite score weight |
| `scoring` | `feasibility_weight` | `0.35` | Composite score weight |
| `scoring` | `impact_weight` | `0.25` | Composite score weight |

### `prompts/agent_prompts.yaml`

Each agent has a `system` prompt (role + rules) and a `user` prompt template with `{placeholder}` variables. Modify prompts here without changing any Python code.

---

## 🔌 MCP Tool System

The system uses **MCP-style tool calling** — Claude autonomously decides when and how to call tools:

```python
# tools/arxiv_tool.py
ARXIV_TOOL_SCHEMA = {
    "name": "search_arxiv",
    "description": "Search arXiv for scientific papers...",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 5}
        }
    }
}
```

The retrieval agent sends all tool schemas to Claude, then:
1. Claude returns `tool_use` blocks autonomously
2. System dispatches to `execute_search_arxiv()`
3. Results are returned as `tool_result` blocks
4. Claude continues until it has enough papers

**Adding a new tool**: Add schema + executor to `TOOL_REGISTRY` in `tools/arxiv_tool.py`. No other changes needed.

---

## 🚀 Future Improvements

1. **Semantic Scholar integration** — richer metadata, citation graphs
2. **PDF full-text parsing** — beyond abstracts using PyMuPDF
3. **Iterative refinement loop** — Critic can trigger re-planning if scores are too low
4. **Parallel retrieval** — run multiple arXiv queries concurrently with asyncio
5. **Vector store persistence** — save/load FAISS index across sessions
6. **Multi-modal support** — parse figures and tables from papers
7. **Email/Slack output** — push experiment plans to external tools
8. **Human-in-the-loop** — pause at CriticAgent for human review before finalizing

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| `ANTHROPIC_API_KEY not set` | Add key to `.env` file |
| `Failed to load embedding model` | Check `EMBEDDING_MODEL_PATH` in `.env` points to downloaded model |
| `arxiv` returns 0 papers | Broaden search query; check internet connectivity |
| `json_parse_failed` in logs | Increase `max_tokens` in `config.yaml`; model output was truncated |
| Langfuse traces not appearing | Verify `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are correct |
| FAISS import error | Run `pip install faiss-cpu` (not `faiss-gpu` unless you have CUDA) |

---

## 📄 License

MIT License — see `LICENSE` file.
