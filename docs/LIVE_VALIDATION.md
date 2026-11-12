# Task #2: live agent integration validation

This test runs the existing Lead and UI Agents against the OpenAI Responses API.
It adds no agents or tools. The only task is the repository's fictional Lantern
Board example; there is no argument for supplying private files or arbitrary tasks.

## Configuration verified before execution

- Python 3.10+; `requirements.txt` pins the official `openai-agents==0.24.0` SDK
  and `python-dotenv==1.2.4`. The live runner rejects an SDK version mismatch.
- The default `gpt-4.1-mini` supports the [Responses API and function
  calling](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
  `OPENAI_MODEL` or `--model` can select another model with those capabilities.
  A local configuration pass does **not** establish live account/model access.
- Both agents use the same explicit `OpenAIResponsesModel`. The SDK's
  [`Agent.as_tool()`](https://openai.github.io/openai-agents-python/tools/#agents-as-tools)
  runs the UI specialist and returns control to the Lead. The Lead requires its
  specialist tool on the first turn, then returns to automatic tool choice.
- The test uses only `https://api.openai.com/v1`. A conflicting
  `OPENAI_BASE_URL` is rejected before sending credentials. Normal environment
  HTTPS proxy/CA settings still apply. No permission or network policy changes
  are made by the test.

## Exact execution steps

Task #2 is on `feature/live-agent-validation`, based on PR #1's
`feature/h2o-prototype`. Its review PR targets that feature branch. Neither PR
should be merged automatically. Use a separate checkout/worktree for concurrent
work rather than switching another chat's active branch.

In a checkout of the Task #2 branch, run:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
python live_test.py
```

The last command must print `"status": "skipped"` and make no API calls, even if
a key is present. Offline test discovery never invokes the paid live runner.

Copy `.env.example` to `.env` in **this checkout**, then set `OPENAI_API_KEY` with
your editor or use your environment's secret settings. The key needs access to
the selected model and API project with available billing/quota. A ChatGPT
subscription or GitHub connection does not supply API credits or an API key.
Never paste the key into chat, task text, command arguments, or version control.
Existing environment variables take precedence over this checkout's `.env`.

```bash
mkdir -p artifacts
python live_test.py --run-live > artifacts/live-validation.json
# Check the exit code immediately: echo $? (PowerShell: $LASTEXITCODE).
python -m json.tool artifacts/live-validation.json
```

Optional model override:

```bash
python live_test.py --run-live --model gpt-4.1-mini
```

Reports go to standard output, so redirect them to save results. `artifacts/` is
gitignored. The report contains the fictional delegated input and model outputs,
SDK/model/Python versions, fixture SHA-256, lifecycle events, tool-call IDs,
response IDs, and reported token usage. It does not include API keys, headers,
or raw API exception bodies. Do not commit credentials or local `.env` files.

## What a pass establishes

The harness observes the real SDK hooks for both agent runs and checks that:

1. The Lead invoked `review_ui` with a tool-call ID.
2. The UI Agent completed with a nonempty review.
3. The corresponding tool completed with that same review and call ID.
4. A subsequent Lead model request contained that review as a tool output.
5. The Lead finished afterward with a nonempty recommendation distinct from the
   specialist review.

A pass reports `status: passed` and all four checks (`configuration`,
`authentication`, `delegation`, `response`) as `passed`. Usually there are three
completed model requests: Lead delegation, UI review, Lead synthesis. Additional
reviews are permitted within the original turn limits.

Inspect `evidence.ui_reviews`, `evidence.received_by_lead`, and `recommendation`
together. Check that the recommendation uses the UI findings, prioritizes changes
within the one-week constraint, supplies acceptance checks, and avoids invented
evidence. The automated checks validate execution and data flow, not the semantic
quality of a nondeterministic recommendation. No extra model grades the result.

## Failures and exit codes

| Status/category | Meaning and next action |
| --- | --- |
| `skipped` / exit 0 | No `--run-live`; no live validation attempted. |
| `blocked`, `authentication` / exit 2 | Missing key; configure it privately. No API call made. |
| `blocked`, `configuration` / exit 2 | Fix Python/dependencies, model setting, API base URL, or missing fictional fixture. No API call made. |
| `failed`, `authentication` / exit 1 | API authentication rejected (401); check or replace the key. |
| `failed`, `configuration` / exit 1 | API rejected model/project access or request configuration (400/403/404). |
| `failed`, `api_limit` / exit 1 | API quota/rate/billing limit (429); inspect the API dashboard before rerunning. |
| `failed`, `connectivity` / exit 1 | Connection/request timeout or overall run timeout. |
| `failed`, `delegation` / exit 1 | Missing/incomplete tool flow, invalid tool behavior, or exhausted turn limit. |
| `failed`, `response` / exit 1 | Empty, incomplete, copied, or otherwise unusable response. |
| `failed`, `api_response` / exit 1 | Another API HTTP failure, such as a server error. |

Reports show partial lifecycle evidence on live failures, including failures
wrapped by the SDK inside the UI tool. Unreached checks remain `not_run`; a local
configuration check is not a claim that live authentication passed. Missing
dependencies produce a direct configuration error with the installation command.

## Billing and limits

Only `--run-live` enables paid calls. OpenAI API input and output tokens are
charged to the API project associated with the key, separately from ChatGPT
subscriptions. Both agents' requests count; the synthesis request includes the
earlier task/tool history. See [current API pricing](https://openai.com/api/pricing/)
and your [API usage dashboard](https://platform.openai.com/usage) for actual costs.

For this test only, output is capped at 1,200 tokens per request, automatic client
and SDK retries are disabled, and orchestration has a 180-second timeout. The
OpenAI client uses a 45-second request timeout, in addition to the existing
runner's 60-second model-call limit and Lead/UI turn limits of four/two.
The test disables tracing and sets Responses API `store=False`; normal provider
data policies still apply. The regular CLI's defaults are unchanged.

These are request/time limits, **not a hard dollar spending cap**. Set API project
billing controls as appropriate. Each rerun can incur more charges. Failed,
truncated, or timed-out runs can still be billable; cancellation does not undo
requests already processed. Reported usage covers completed model responses and
may omit usage from failed or timed-out requests. An incomplete response fails
validation instead of being accepted as a full review; inspect the report before
deciding whether to change the output limit or rerun.

## Development validation record

The original nine offline tests remain unchanged. Additional offline tests cover
the opt-in gate, missing-key and configuration blocks, wrapped API error
categories/redaction, SDK lifecycle evidence, copied/truncated responses, and
the test-only client limits. These use scripted model responses or mocked API
errors; they do not establish live model correctness.

Development results: **21 offline tests passed** (nine original, twelve new),
dependency consistency passed, and all Python files passed a Python 3.10 syntax
check. Execution used Python 3.12.14; a Python 3.10 runtime was not exercised.

The development environment initially had no `OPENAI_API_KEY` or project `.env`.
The opt-in command therefore reported `blocked/authentication` before any API
request (exit code 2, no API usage from the attempt). A live pass must be recorded
separately after credentials are configured;
an offline pass or a missing-key block must never be reported as live success.
