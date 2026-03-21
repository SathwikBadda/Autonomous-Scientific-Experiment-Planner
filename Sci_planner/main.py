"""
main.py - Entrypoint for running the API server or CLI pipeline.

Usage:
  # Run API server
  python main.py serve

  # Run pipeline from CLI
  python main.py run "Improve transformer efficiency for long context"

  # Run pipeline with structured input
  python main.py run --domain NLP --task "Text Generation" --constraint "Low compute"
"""
from __future__ import annotations

import argparse
import json
import sys
import uvicorn

from config.settings import config
from utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def serve():
    """Start the FastAPI server."""
    api_cfg = config["api"]
    logger.info(
        "starting_server",
        host=api_cfg["host"],
        port=api_cfg["port"],
    )
    uvicorn.run(
        "api.app:app",
        host=api_cfg["host"],
        port=api_cfg["port"],
        reload=api_cfg.get("reload", False),
        workers=api_cfg.get("workers", 1),
        log_level=config["app"]["log_level"].lower(),
    )


def run_cli(research_input):
    """Run the pipeline directly from CLI and print JSON output."""
    from agents.workflow import run_pipeline
    from utils.output_formatter import format_final_output

    logger.info("cli_run_start", input=str(research_input)[:200])
    state = run_pipeline(research_input)
    output = format_final_output(state)
    print(json.dumps(output, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Scientific Experiment Planner"
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve command
    subparsers.add_parser("serve", help="Start the FastAPI server")

    # run command
    run_parser = subparsers.add_parser("run", help="Run pipeline from CLI")
    run_parser.add_argument(
        "idea",
        nargs="?",
        help="Research idea as plain text",
    )
    run_parser.add_argument("--domain", help="Research domain (for structured input)")
    run_parser.add_argument("--task", help="Specific task (for structured input)")
    run_parser.add_argument("--constraint", help="Constraint (for structured input)")

    args = parser.parse_args()

    if args.command == "serve":
        serve()
    elif args.command == "run":
        if args.domain:
            research_input = {
                "domain": args.domain,
                "task": args.task or "",
                "constraint": args.constraint or "",
            }
        elif args.idea:
            research_input = args.idea
        else:
            parser.print_help()
            sys.exit(1)
        run_cli(research_input)
    else:
        # Default: serve
        serve()


if __name__ == "__main__":
    main()
