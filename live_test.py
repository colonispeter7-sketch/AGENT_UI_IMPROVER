"""Opt-in live validation. Prints a JSON report; uses only the fictional sample task."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

try:
    from agents import ModelRetrySettings, ModelSettings, RunHooks
    from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
    from agents.models.openai_responses import OpenAIResponsesModel
    from dotenv import load_dotenv
    from openai import (
        APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI,
        AuthenticationError, BadRequestError, NotFoundError, PermissionDeniedError,
        RateLimitError,
    )

    import h2o
except ImportError:
    raise SystemExit("FAIL [configuration]: install dependencies with python -m pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent
API_BASE = "https://api.openai.com/v1"
EXPECTED_SDK = "0.24.0"
MAX_OUTPUT_TOKENS = 1200
RUN_TIMEOUT = 180


class Evidence(RunHooks):
    """Observe both agents without giving either agent additional tools."""

    def __init__(self):
        self.events = []
        self.calls = []
        self.outputs = []
        self.received_by_lead = []
        self.ui_reviews = []
        self.requests = []

    def event(self, kind, agent):
        self.events.append({"event": kind, "agent": agent.name})

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        self.event("model_request", agent)
        if agent.name == "Lead Agent":
            for item in input_items:
                if isinstance(item, dict) and item.get("type") == "function_call_output":
                    self.received_by_lead.append({
                        "call_id": item.get("call_id"), "output": item.get("output"),
                        "ui_completed": bool(self.ui_reviews),
                    })

    async def on_llm_end(self, context, agent, response):
        self.event("model_response", agent)
        incomplete = any(getattr(item, "status", None) == "incomplete" for item in response.output)
        self.requests.append({
            "agent": agent.name, "response_id": response.response_id,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "incomplete": incomplete,
        })
        if incomplete:
            raise h2o.ResponseError("A model response was incomplete, possibly due to the output limit.")

    async def on_tool_start(self, context, agent, tool):
        self.event("tool_start", agent)
        self.calls.append({
            "agent": agent.name, "tool": tool.name,
            "call_id": getattr(context, "tool_call_id", None),
            "arguments": getattr(context, "tool_arguments", None),
        })

    async def on_tool_end(self, context, agent, tool, result):
        self.event("tool_end", agent)
        self.outputs.append({
            "call_id": getattr(context, "tool_call_id", None), "output": result,
        })

    async def on_agent_end(self, context, agent, output):
        self.event("agent_completed", agent)
        if agent.name == "UI Agent":
            self.ui_reviews.append(output)

    def validate(self, result):
        if not self.calls or len(self.calls) != len(self.outputs):
            raise h2o.DelegationError("Specialist tool calls did not all complete.")
        if tuple(self.ui_reviews) != result.ui_reviews:
            raise h2o.DelegationError("Specialist lifecycle does not match returned reviews.")
        for call, output in zip(self.calls, self.outputs):
            if (call["agent"] != "Lead Agent" or call["tool"] != "review_ui"
                    or not call["call_id"] or call["call_id"] != output["call_id"]
                    or output["output"] not in result.ui_reviews
                    or not any(item["call_id"] == call["call_id"]
                               and item["output"] == output["output"]
                               and item["ui_completed"] for item in self.received_by_lead)):
                raise h2o.DelegationError("UI findings were not delivered to Lead synthesis.")
        if not self.events or self.events[-1] != {
            "event": "agent_completed", "agent": "Lead Agent"
        }:
            raise h2o.DelegationError("Lead did not resume and finish after the specialist.")
        if (not result.recommendation.strip()
                or result.recommendation.strip() in {text.strip() for text in result.ui_reviews}):
            raise h2o.ResponseError("Lead did not produce a distinct recommendation.")

    def report(self):
        return {
            "events": self.events, "tool_calls": self.calls, "tool_outputs": self.outputs,
            "received_by_lead": self.received_by_lead, "ui_reviews": self.ui_reviews,
            "requests": self.requests,
            "input_tokens": sum(item["input_tokens"] for item in self.requests),
            "output_tokens": sum(item["output_tokens"] for item in self.requests),
        }


def classify_error(error):
    """Inspect wrapped specialist errors without printing API bodies or secrets."""
    chain = []
    while error is not None and id(error) not in {id(item) for item in chain}:
        chain.append(error)
        error = error.__cause__ or error.__context__
    categories = [
        (AuthenticationError, "authentication", "API authentication failed. Check OPENAI_API_KEY."),
        ((PermissionDeniedError, NotFoundError, BadRequestError), "configuration",
         "Check model availability, project access, and Responses API tool-call support."),
        (RateLimitError, "api_limit", "Check API quota, billing, and rate limits before retrying."),
        ((APITimeoutError, APIConnectionError, TimeoutError, asyncio.TimeoutError), "connectivity",
         "The API or overall run timed out, or could not connect. Check connectivity."),
        (h2o.ResponseError, "response", "An agent returned an empty, incomplete, or copied response. Inspect the evidence."),
        ((h2o.DelegationError, ModelBehaviorError, MaxTurnsExceeded), "delegation",
         "The specialist tool flow did not complete within its limits. Inspect the event evidence."),
        (APIStatusError, "api_response", "The API returned an unexpected HTTP status."),
    ]
    for types, category, message in categories:
        for cause in chain:
            if isinstance(cause, types):
                return category, message, type(cause).__name__
    return "response", "Unexpected SDK or response failure; no validated result.", type(chain[0]).__name__


async def execute(model, evidence, task):
    # Explicit official endpoint; do not inherit OPENAI_BASE_URL.
    # Normal HTTPS proxy/CA settings still apply.
    async with AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"], base_url=API_BASE, max_retries=0, timeout=45
    ) as client:
        return await asyncio.wait_for(
            h2o.review(
                task,
                OpenAIResponsesModel(model=model, openai_client=client),
                hooks=evidence,
                model_settings=ModelSettings(
                    max_tokens=MAX_OUTPUT_TOKENS, store=False,
                    retry=ModelRetrySettings(max_retries=0),
                ),
            ),
            timeout=RUN_TIMEOUT,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true", help="Allow billable OpenAI API requests")
    parser.add_argument("--model", help="Tool-capable Responses API model; overrides OPENAI_MODEL")
    args = parser.parse_args(argv)
    report = {"status": "skipped", "message": "No API calls. Add --run-live to opt in to API billing."}
    if not args.run_live:
        print(json.dumps(report, indent=2))
        return 0

    try:
        load_dotenv(ROOT / ".env")
    except (OSError, UnicodeError):
        print(json.dumps({"status": "blocked", "category": "configuration",
                          "message": "Cannot read this checkout's UTF-8 .env file."}, indent=2))
        return 2
    model = args.model if args.model is not None else os.getenv("OPENAI_MODEL", h2o.DEFAULT_MODEL)
    checks = dict.fromkeys(("configuration", "authentication", "delegation", "response"), "not_run")
    report = {
        "status": "blocked", "model": model, "sdk_version": importlib.metadata.version("openai-agents"),
        "python": sys.version.split()[0], "checks": checks,
        "fixture": "examples/sample_task.txt", "max_output_tokens_per_request": MAX_OUTPUT_TOKENS,
        "overall_timeout_seconds": RUN_TIMEOUT, "automatic_retries": 0,
    }

    def finish(code, category, message):
        report.update(category=category, message=message)
        print(json.dumps(report, indent=2))
        return code

    if (sys.version_info < (3, 10) or report["sdk_version"] != EXPECTED_SDK
            or not model.strip()
            or os.getenv("OPENAI_BASE_URL", API_BASE).rstrip("/") != API_BASE):
        checks["configuration"] = "failed"
        return finish(2, "configuration", "Use Python 3.10+, requirements.txt, a nonempty model, and the official API base URL.")
    checks["configuration"] = "passed"
    if not os.getenv("OPENAI_API_KEY", "").strip():
        checks["authentication"] = "blocked"
        return finish(2, "authentication", "Set OPENAI_API_KEY in your environment or local .env; never paste it in chat.")

    evidence = Evidence()
    try:
        fixture = ROOT / "examples/sample_task.txt"
        task_bytes = fixture.read_bytes()
        task = task_bytes.decode("utf-8")
        report["fixture_sha256"] = hashlib.sha256(task_bytes).hexdigest()
    except (OSError, UnicodeError):
        checks["configuration"] = "failed"
        return finish(2, "configuration", "Cannot read the fixed UTF-8 fictional sample task.")

    try:
        result = asyncio.run(execute(model, evidence, task))
        report["recommendation"] = result.recommendation
        evidence.validate(result)
    except Exception as error:
        report["status"] = "failed"
        category, message, error_type = classify_error(error)
        report["exception_type"] = error_type
        report["evidence"] = evidence.report()
        if evidence.requests:
            checks["authentication"] = "passed"
        if category in checks:
            checks[category] = "failed"
        return finish(1, category, message)

    checks.update(dict.fromkeys(checks, "passed"))
    report.update(
        status="passed", evidence=evidence.report(), recommendation=result.recommendation,
        manual_quality_review_required=True,
    )
    return finish(0, "success", "Live delegation and Lead synthesis flow verified. Inspect recommendation quality manually.")


if __name__ == "__main__":
    raise SystemExit(main())
