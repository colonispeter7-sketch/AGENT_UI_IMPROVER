# H2O: two-agent development review prototype

A minimal manager-style workflow using the official [OpenAI Agents SDK for
Python](https://openai.github.io/openai-agents-python/) (`openai-agents` 0.24.0).
Give it a development objective and a frontend application description; it returns
a prioritized recommendation. It reviews descriptions; it does not implement code.

## Architecture

```text
CLI objective + application description
                 |
                 v
             Lead Agent
                 |
                 | review_ui tool (UI Agent.as_tool)
                 v
              UI Agent
                 |
                 | navigation, usability, accessibility, design findings
                 v
             Lead Agent
                 |
                 v
       final recommendation
```

The Lead owns the conversation and synthesizes the specialist's findings. The UI
Agent is a tool, not a handoff: it receives the task the Lead delegates and returns
its review to the Lead. There are exactly two agents and one tool. Neither agent
has browser, shell, file, repository, or other external-service tools.

The Lead's first model turn requires a tool call; the SDK resets that requirement
after the call so the Lead can answer. Completed, nonempty specialist reviews are
recorded through `custom_output_extractor`. The run fails if there is no completed
review or final recommendation. Specialist failures propagate rather than being
treated as review findings. The Lead is limited to four turns, each specialist run
to two, with a 60-second timeout per model request (SDK retries may add time).

`h2o.py` contains the agent definitions, orchestration, and standard-library CLI.
`tests/test_smoke.py` uses `unittest`; there is no app framework, database, server,
or persistent conversation storage.

## Setup and run

Requires Python **3.10+**, an OpenAI API key, and access to a model supporting tool
calls. Run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in your local `.env` using your editor, or export it in your
shell. Do not paste credentials into task descriptions or commit them. The CLI
loads only the project's `.env`; existing environment variables take precedence.
The same model is used for both agents. `OPENAI_MODEL` defaults to `gpt-4.1-mini`;
`--model` overrides it.

```bash
python h2o.py --task-file examples/sample_task.txt
python h2o.py --objective "Review the fictional Kite app: help visitors find classes. Its mobile menu uses unlabeled icons. Recommend a one-day improvement plan."
python h2o.py --task-file examples/sample_task.txt --model gpt-4.1-mini
python h2o.py --help
```

Use exactly one of `--objective` or `--task-file`. A file must be UTF-8 and is read
only when explicitly supplied. The included Lantern Board task is entirely
fictional demonstration data. Do not supply HHC files or confidential content.
Application descriptions are sent to the OpenAI API; live runs incur API usage
costs. SDK tracing is explicitly disabled for both agents. No separate trace
upload or local transcript is created.

## Verify delegation

Run the offline smoke test, which requires neither an API key nor network calls:

```bash
python -m unittest discover -s tests -v
```

The test uses the **real SDK Runner and `Agent.as_tool()`**, substituting only the
model responses. It scripts a Lead tool call, a UI review, and a Lead response. It
checks that the UI received the task, the review returned as a tool output to the
Lead, and the Lead resumed to synthesize. It also checks missing delegation,
specialist failure, empty reviews/objectives, missing credentials, and CLI output.
This validates orchestration, not live model quality or API access.

For a live check, configure your key and run the sample task. Standard error should
contain:

```text
Delegation verified: UI Agent completed 1 review(s).
```

That line is emitted only after a real specialist review and final recommendation
complete; the count may be higher if the Lead requests additional reviews.
Standard output contains the Lead's recommendation, attributing specialist
findings to the UI Agent. Expect prioritized findings about discoverability,
booking friction, and accessibility; wording and prioritization vary by model.
You can save only the recommendation with:

```bash
python h2o.py --task-file examples/sample_task.txt > recommendation.txt
```

A zero exit code means the workflow completed. Invalid arguments or a missing key
exit with code 2; file, SDK, or API failures exit with code 1. Errors identify their
type without printing request bodies or credentials. Check key validity, quota,
model access, and connectivity for an API failure.
