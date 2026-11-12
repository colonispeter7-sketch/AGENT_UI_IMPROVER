"""Offline smoke tests: use the real SDK runner with a scripted model boundary."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import AsyncMock, patch

from agents import Model, ModelResponse, Usage, UserError
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage

import h2o

TASK = "Improve the fictional Lantern Board mobile navigation. The menu is unlabeled."
FINDING = "P1: Label the menu. Acceptance: the button has an accessible name."
RECOMMENDATION = "UI Agent recommends labeling the menu; implement this first."


def message(text):
    return ResponseOutputMessage(
        id="message", type="message", role="assistant", status="completed",
        content=[{"type": "output_text", "text": text, "annotations": []}],
    )


class ScriptedModel(Model):
    """Only the model is substituted; agent/tool execution is the SDK's."""

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    async def get_response(self, system_instructions, input, model_settings, tools, **kwargs):
        self.calls.append((system_instructions, input, model_settings, tools))
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return ModelResponse(output=[output], usage=Usage(), response_id=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError("This prototype uses non-streaming runs.")


def delegation():
    return ResponseFunctionToolCall(
        id="tool_call", type="function_call", call_id="ui_review_1",
        name="review_ui", arguments=json.dumps({"input": TASK}),
    )


class OrchestrationSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_lead_delegates_to_ui_then_synthesizes(self):
        model = ScriptedModel([delegation(), message(FINDING), message(RECOMMENDATION)])
        result = await h2o.review(TASK, model)

        self.assertEqual(result.recommendation, RECOMMENDATION)
        self.assertEqual(result.ui_reviews, (FINDING,))
        self.assertEqual(len(model.calls), 3)
        lead, ui, synthesis = model.calls
        self.assertIn("Lead Agent", lead[0])
        self.assertEqual([tool.name for tool in lead[3]], ["review_ui"])
        self.assertEqual(lead[2].tool_choice, "required")
        self.assertIn("UI Agent", ui[0])
        self.assertEqual(ui[3], [])  # Specialist has no tools or filesystem access.
        self.assertIn(TASK, str(ui[1]))
        self.assertIn("Lead Agent", synthesis[0])
        self.assertNotEqual(synthesis[2].tool_choice, "required")
        tool_outputs = [item for item in synthesis[1]
                        if isinstance(item, dict) and item.get("type") == "function_call_output"]
        self.assertEqual(tool_outputs[0]["call_id"], "ui_review_1")
        self.assertEqual(tool_outputs[0]["output"], FINDING)

    async def test_missing_delegation_is_rejected(self):
        model = ScriptedModel([message("I skipped the review.")])
        with self.assertRaisesRegex(RuntimeError, "without a completed UI Agent review"):
            await h2o.review(TASK, model)

    async def test_specialist_failure_propagates(self):
        model = ScriptedModel([delegation(), RuntimeError("Specialist failed")])
        # The SDK wraps tool exceptions in UserError while preserving the cause.
        with self.assertRaisesRegex(UserError, "Specialist failed"):
            await h2o.review(TASK, model)
        self.assertEqual(len(model.calls), 2)  # No synthesis of a failed review.

    async def test_empty_specialist_review_is_rejected(self):
        model = ScriptedModel([delegation(), message(" ")])
        with self.assertRaisesRegex(UserError, "empty review"):
            await h2o.review(TASK, model)

    async def test_empty_lead_recommendation_is_rejected(self):
        model = ScriptedModel([delegation(), message(FINDING), message(" ")])
        with self.assertRaisesRegex(RuntimeError, "empty recommendation"):
            await h2o.review(TASK, model)

    async def test_empty_objective_is_rejected_before_model_call(self):
        model = ScriptedModel([])
        with self.assertRaises(ValueError):
            await h2o.review(" \n", model)
        self.assertEqual(model.calls, [])


class CliSmokeTests(unittest.TestCase):
    @patch("h2o.load_dotenv")
    def test_missing_credentials_has_actionable_error(self, _load):
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}), redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit) as exit_status:
                h2o.main(["--objective", TASK])
        self.assertEqual(exit_status.exception.code, 2)
        self.assertIn("Set OPENAI_API_KEY", err.getvalue())

    @patch("h2o.load_dotenv")
    def test_cli_reads_sample_task_and_reports_completed_delegation(self, _load):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "offline-test-placeholder"}), \
             patch("h2o.review", new_callable=AsyncMock) as run, \
             redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            run.return_value = h2o.Review(RECOMMENDATION, (FINDING,))
            status = h2o.main(["--task-file", "examples/sample_task.txt"])
        self.assertEqual(status, 0)
        self.assertIn("Fictional demonstration", run.call_args.args[0])
        self.assertEqual(out.getvalue().strip(), RECOMMENDATION)
        self.assertIn("UI Agent completed 1 review(s)", err.getvalue())

    @patch("h2o.load_dotenv")
    def test_cli_failure_reports_no_success_or_sensitive_error_body(self, _load):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "offline-test-placeholder"}), \
             patch("h2o.review", new_callable=AsyncMock) as run, \
             redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            run.side_effect = OSError("request body should not be printed")
            status = h2o.main(["--objective", TASK])
        self.assertEqual(status, 1)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("Review failed (OSError)", err.getvalue())
        self.assertNotIn("request body", err.getvalue())
        self.assertNotIn("Delegation verified", err.getvalue())


if __name__ == "__main__":
    unittest.main()
