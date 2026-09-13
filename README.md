# fluiq-worker-evaluator

LLM-as-judge and trajectory-level agentic evaluation. Consumes evaluation
requests from Kafka, scores a run, writes results to ClickHouse.

> **Status: archived.** Part of [Fluiq](https://github.com/fluiq-AI), which ran
> from 10 April to September 2026 and never found customers. The hosted service
> is shut down. MIT, unmaintained, fork freely.

This is the most interesting repo of the three. Most evaluation tooling scores
the final answer. This scores the path: which tools the agent chose, whether the
trajectory made sense, what the retrieval actually supported, and how multiple
agents coordinated.

| | |
|---|---|
| **Consumes** | `KAFKA_EVAL_TOPIC` as group `KAFKA_EVAL_GROUP_ID` |
| **Writes** | ClickHouse `evaluations`; judge prompts and rubrics from Postgres |

## The layers

[`jobs/agentic/orchestrator.py`](jobs/agentic/orchestrator.py) composes four
layers over a normalized `AgentRun`. Depth is chosen per run, so a streaming
evaluation can stay cheap while an offline or CI run goes deep.

| Layer | What it does | Depth |
|---|---|---|
| **L1 deterministic** | Structural checks, no LLM, no cost | always |
| **L2 tool selection** | Did it pick the right tool for the step? Plus retrieval quality | always |
| **L3 trajectory** | Did the sequence of steps make sense as a whole? | `standard`, `deep` |
| **L4 multi-agent panel** | Re-judges L2 and L3 with a panel of models | `deep`, gated |

Depth values are `fast` (L1+L2), `standard` (+L3) and `deep` (+L4).

Each metric is persisted with the layer it came from on the ClickHouse `layer`
column — `tool_selection`, `retrieval`, `trajectory`, `coordination` — so the UI
can show which part of the agent's behaviour scored badly rather than one
undifferentiated number.

**The panel is gated by default.** `EVAL_PANEL_MODE=gated` only convenes the
multi-model panel when the primary judge's score falls within
`EVAL_PANEL_GATE_MARGIN` (default 0.12) of the threshold — the band where a
single judge is least trustworthy and most likely to flip. Everywhere else, one
judge is enough and three models is three times the cost for no information.
Set `EVAL_PANEL_MODE=always` to convene it every time.

## Configuration

Read in [`config.py`](config.py). Judge settings:

```
EVAL_JUDGE_PROVIDER=anthropic
EVAL_JUDGE_MODEL=claude-haiku-4-5-20251001
EVAL_JUDGE_THRESHOLD=0.7
EVAL_AGENT_DEPTH=standard          # fast | standard | deep
EVAL_PANEL_MODE=gated              # gated | always | off
EVAL_PANEL_GATE_MARGIN=0.12
EVAL_PANEL_MEMBERS=                # comma-separated model ids; empty = default trio
EVAL_JUDGE_CACHE=1
EVAL_JUDGE_CACHE_TTL=3600
EVAL_JUDGE_CACHE_MAX=1000
ANTHROPIC_API_KEY=sk-ant-...
```

Kafka, ClickHouse and Postgres settings match the other services — see
[fluiq-api/.env.example](https://github.com/fluiq-AI/fluiq-api/blob/main/.env.example).
Two consumer settings matter here specifically:

```
KAFKA_MAX_POLL_INTERVAL_MS=900000   # 15 min
KAFKA_SESSION_TIMEOUT_MS=45000
```

A deep evaluation with a panel can take minutes. At the default poll interval
Kafka concludes the consumer is dead and rebalances mid-evaluation, which
reassigns the partition and evaluates the same run twice.

## Running

Bring up the data stores from
[fluiq-api](https://github.com/fluiq-AI/fluiq-api)'s compose file, then:

```bash
docker build -t fluiq-evaluator .
docker run --network fluiq-ai --env-file .env.development fluiq-evaluator
```

Or directly:

```bash
pip install -r requirements.txt
python app.py
```

## Tests

```bash
python -m pytest tests/
```

`tests/` covers the agentic layers offline against fixture runs — no API key and
no network needed for that suite.

## Notable decisions

**Judge failure fails open.** If the model errors, times out, or returns
something unparseable, the span is recorded unscored and nothing blocks. An
evaluation tool that takes production down when Anthropic has a bad minute is
worse than no evaluation tool. This was a real regression once and is the thing
most worth not undoing.

**Seek-back retry needs explicit dedupe.** Like the tracer, a transient failure
rewinds to the message's own offset rather than committing past it. Unlike the
tracer, evaluation has side effects, so `_already_processed` tracks
`(topic, partition, offset)` to keep the retry path from writing a duplicate
evaluation for a run that partially succeeded.

**Memory is the binding constraint, not CPU.** The image carries torch and spaCy;
their baseline plus glibc malloc arenas put the RSS floor above a 2 GB task
before a single message is consumed. If you run this in a container, size it at
4 GB or trim the dependencies.

## Licence

MIT. See [LICENSE](LICENSE).
