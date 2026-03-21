"""
api/app.py - FastAPI REST layer for the Autonomous Scientific Experiment Planner.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agents.workflow import run_pipeline
from config.settings import config
from utils.logger import get_logger
from utils.output_formatter import format_final_output

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Pydantic request/response models
# ─────────────────────────────────────────────

class PlainInput(BaseModel):
    research_idea: str = Field(
        ...,
        description="Free-text research idea or question.",
        example="Improve transformer efficiency for long context understanding",
    )


class StructuredInput(BaseModel):
    domain: str = Field(..., example="NLP")
    task: str = Field(..., example="Text Generation")
    constraint: str = Field(..., example="Low compute")
    additional_context: str | None = Field(None, example="Focus on low-resource languages")


class PlanRequest(BaseModel):
    """Accepts either plain text or structured input."""
    plain_input: str | None = Field(None, description="Free-text research idea")
    structured_input: StructuredInput | None = Field(None, description="Structured research spec")

    def to_research_input(self) -> Any:
        if self.structured_input:
            return self.structured_input.model_dump(exclude_none=True)
        return self.plain_input or ""


class ExperimentPlanResponse(BaseModel):
    request_id: str
    processing_time_s: float
    problem_statement: str
    literature_summary: str
    identified_gaps: list[str]
    hypotheses: list[str]
    experiment_plan: dict
    datasets: list[str]
    expected_outcomes: str
    novelty_score: float
    feasibility_score: float
    impact_score: float
    composite_score: float
    overall_recommendation: str
    critique_summary: str
    papers_retrieved: int
    agent_trace: list[str]
    trace_id: str | None
    error: str | None


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(
    title="Autonomous Scientific Experiment Planner",
    description=(
        "Multi-agent AI system that analyzes scientific literature, identifies "
        "research gaps, generates hypotheses, and proposes detailed experiment plans."
    ),
    version=config["app"]["version"],
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Middleware: request logging
# ─────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round(time.time() - start, 3)
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        elapsed_s=elapsed,
    )
    return response


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": config["app"]["name"],
        "version": config["app"]["version"],
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "version": config["app"]["version"]}


@app.post(
    "/plan",
    response_model=ExperimentPlanResponse,
    tags=["Planning"],
    summary="Generate an autonomous experiment plan from a research idea",
)
async def create_experiment_plan(request: PlanRequest):
    """
    Main endpoint. Accepts a research idea (plain or structured) and
    returns a complete, AI-generated experiment plan with scores.

    - **plain_input**: e.g. "Improve transformer efficiency for long context"
    - **structured_input**: domain + task + constraint JSON
    """
    request_id = str(uuid.uuid4())[:8]
    research_input = request.to_research_input()

    if not research_input:
        raise HTTPException(status_code=422, detail="Provide plain_input or structured_input")

    logger.info("plan_request", request_id=request_id, input=str(research_input)[:200])
    start = time.time()

    try:
        final_state = run_pipeline(research_input)
        output = format_final_output(final_state)
        elapsed = round(time.time() - start, 2)

        logger.info(
            "plan_complete",
            request_id=request_id,
            elapsed_s=elapsed,
            novelty=output.get("novelty_score"),
            feasibility=output.get("feasibility_score"),
        )

        return ExperimentPlanResponse(
            request_id=request_id,
            processing_time_s=elapsed,
            **{k: v for k, v in output.items() if k in ExperimentPlanResponse.model_fields},
        )

    except Exception as exc:
        elapsed = round(time.time() - start, 2)
        logger.error("plan_failed", request_id=request_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/plan/plain",
    response_model=ExperimentPlanResponse,
    tags=["Planning"],
    summary="Quick endpoint for plain-text research ideas",
)
async def plan_from_plain_text(body: PlainInput):
    """Convenience endpoint: just pass a research idea string."""
    return await create_experiment_plan(
        PlanRequest(plain_input=body.research_idea)
    )


@app.post(
    "/plan/structured",
    response_model=ExperimentPlanResponse,
    tags=["Planning"],
    summary="Endpoint for structured research specifications",
)
async def plan_from_structured(body: StructuredInput):
    """Convenience endpoint: pass domain, task, and constraint."""
    return await create_experiment_plan(
        PlanRequest(structured_input=body)
    )
