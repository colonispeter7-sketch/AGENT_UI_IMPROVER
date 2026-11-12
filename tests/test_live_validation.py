"""Offline coverage of the opt-in live-test harness; never calls the API."""

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, Mock, patch

from agents import ModelRetrySettings, ModelSettings, UserError
from openai import (
    APIConnectionError, AuthenticationError, BadRequestError, InternalServerError,
    NotFoundError, PermissionDeniedError, RateLimitError,
)

import h2o
import live_test
from test_smoke import FINDING, RECOMMENDATION, TASK, ScriptedModel, delegation, message


class EvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_sdk_emits_delegation_and_synthesis_evidence(self):
        model = ScriptedModel([delegation(), message(FINDING), message(RECOMMENDATION)])
        evidence = live_test.Evidence()
        settings = ModelSettings(max_tokens=1200, store=False, retry=ModelRetrySettings(max_retries=0))
        result = await h2o.review(TASK, model, hooks=evidence, model_settings=settings)
        evidence.validate(result)
        self.assertEqual([item["agent"] for item in evidence.requests],
                         ["Lead Agent", "UI Agent", "Lead Agent"])
        self.assertEqual(evidence.outputs[0]["output"], FINDING)
        self.assertEqual(evidence.received_by_lead[0]["output"], FINDING)
        self.assertTrue(all(call[2].max_tokens == 1200 for call in model.calls))
        self.assertTrue(all(call[2].store is False for call in model.calls))
        self.assertTrue(all(call[2].retry.max_retries == 0 for call in model.calls))
        evidence.received_by_lead.clear()
        with self.assertRaises(h2o.DelegationError):
            evidence.validate(result)

    async def test_copying_ui_review_does_not_count_as_synthesis(self):
        model = ScriptedModel([delegation(), message(FINDING), message(FINDING)])
        evidence = live_test.Evidence()
        result = await h2o.review(TASK, model, hooks=evidence)
        with self.assertRaises(h2o.ResponseError):
            evidence.validate(result)

    async def test_incomplete_model_response_fails(self):
        incomplete = message(FINDING)
        incomplete.status = "incomplete"
        model = ScriptedModel([delegation(), incomplete])
        with self.assertRaisesRegex(UserError, "incomplete"):
            await h2o.review(TASK, model, hooks=live_test.Evidence())

    async def test_live_client_limits_and_official_endpoint(self):
        client = Mock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "offline-placeholder"}), \
             patch("live_test.AsyncOpenAI", return_value=client) as constructor, \
             patch("live_test.OpenAIResponsesModel", return_value="model") as model, \
             patch("h2o.review", new_callable=AsyncMock) as run:
            await live_test.execute("gpt-4.1-mini", live_test.Evidence(), TASK)
        self.assertEqual(constructor.call_args.kwargs["base_url"], live_test.API_BASE)
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
        self.assertEqual(constructor.call_args.kwargs["timeout"], 45)
        self.assertEqual(model.call_args.kwargs["model"], "gpt-4.1-mini")
        settings = run.call_args.kwargs["model_settings"]
        self.assertEqual(settings.retry.max_retries, 0)
        self.assertEqual(settings.max_tokens, live_test.MAX_OUTPUT_TOKENS)
        self.assertFalse(settings.store)
        self.assertEqual(run.call_args.args[0], TASK)


@patch("live_test.load_dotenv")
class LiveCliTests(unittest.TestCase):
    def call(self, argv):
        with redirect_stdout(io.StringIO()) as stream:
            code = live_test.main(argv)
        return code, json.loads(stream.getvalue())

    def test_default_skips_even_with_a_key(self, _load):
        with patch("live_test.execute", new_callable=AsyncMock) as execute, \
             patch.dict("os.environ", {"OPENAI_API_KEY": "offline-placeholder"}):
            code, report = self.call([])
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "skipped")
        execute.assert_not_called()

    def test_opt_in_without_key_is_blocked(self, _load):
        with patch.dict("os.environ", {}, clear=True), \
             patch("live_test.execute", new_callable=AsyncMock) as execute:
            code, report = self.call(["--run-live"])
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["checks"]["authentication"], "blocked")
        execute.assert_not_called()

    def test_bad_configuration_is_blocked_before_api(self, _load):
        for variables in ({"OPENAI_MODEL": " "}, {"OPENAI_BASE_URL": "https://example.invalid/v1"}):
            with self.subTest(variables=variables), patch.dict("os.environ", variables, clear=True), \
                 patch("live_test.execute", new_callable=AsyncMock) as execute:
                code, report = self.call(["--run-live"])
            self.assertEqual(code, 2)
            self.assertEqual(report["category"], "configuration")
            execute.assert_not_called()

    def test_sdk_mismatch_is_blocked_before_api(self, _load):
        with patch("live_test.importlib.metadata.version", return_value="0.0.0"), \
             patch("live_test.execute", new_callable=AsyncMock) as execute:
            code, report = self.call(["--run-live"])
        self.assertEqual(code, 2)
        self.assertEqual(report["category"], "configuration")
        execute.assert_not_called()

    def test_unreadable_env_is_a_configuration_block(self, load):
        load.side_effect = UnicodeError("sensitive file detail")
        code, report = self.call(["--run-live"])
        self.assertEqual(code, 2)
        self.assertEqual(report["category"], "configuration")
        self.assertNotIn("sensitive", json.dumps(report))

    def test_success_report_includes_reviews_recommendation_and_usage(self, _load):
        async def offline(model, evidence, task):
            self.assertIn("Fictional demonstration", task)
            scripted = ScriptedModel([delegation(), message(FINDING), message(RECOMMENDATION)])
            return await h2o.review(task, scripted, hooks=evidence)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "offline-placeholder"}, clear=True), \
             patch("live_test.execute", side_effect=offline):
            code, report = self.call(["--run-live"])
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(all(value == "passed" for value in report["checks"].values()))
        self.assertEqual(report["recommendation"], RECOMMENDATION)
        self.assertEqual(report["evidence"]["ui_reviews"], [FINDING])
        self.assertEqual(len(report["evidence"]["requests"]), 3)
        self.assertTrue(report["manual_quality_review_required"])

    def test_api_error_categories_and_nested_cause_redaction(self, _load):
        cases = [
            (AuthenticationError, 401, "authentication"),
            (PermissionDeniedError, 403, "configuration"),
            (NotFoundError, 404, "configuration"),
            (BadRequestError, 400, "configuration"),
            (RateLimitError, 429, "api_limit"),
            (InternalServerError, 500, "api_response"),
        ]
        for error_class, status, category in cases:
            response = Mock(status_code=status, request=Mock())
            error = UserError("sensitive outer message")
            error.__cause__ = error_class("sensitive API body", response=response, body=None)
            with self.subTest(status=status), \
                 patch.dict("os.environ", {"OPENAI_API_KEY": "offline-placeholder"}, clear=True), \
                 patch("live_test.execute", new_callable=AsyncMock, side_effect=error):
                code, report = self.call(["--run-live"])
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["category"], category)
            self.assertNotIn("sensitive", json.dumps(report))
            self.assertNotIn("offline-placeholder", json.dumps(report))

    def test_connectivity_delegation_and_response_classification(self, _load):
        for error, category in [
            (TimeoutError(), "connectivity"),
            (APIConnectionError(request=Mock()), "connectivity"),
            (h2o.DelegationError(), "delegation"),
            (h2o.ResponseError(), "response"),
        ]:
            with self.subTest(category=category):
                self.assertEqual(live_test.classify_error(error)[0], category)


if __name__ == "__main__":
    unittest.main()
