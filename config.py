import os
import ssl
from dotenv import load_dotenv

load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_TRACE_TOPIC = os.getenv("KAFKA_TRACE_TOPIC")
KAFKA_TRACE_PERSISTED_TOPIC = os.getenv("KAFKA_TRACE_PERSISTED_TOPIC")
# PLAINTEXT (local docker) | SASL_SSL (AWS MSK SASL/SCRAM)
KAFKA_SECURITY_PROTOCOL=os.getenv("KAFKA_SECURITY_PROTOCOL")
KAFKA_SASL_MECHANISM=os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-512")
KAFKA_SASL_USERNAME=os.getenv("KAFKA_SASL_USERNAME")
KAFKA_SASL_PASSWORD=os.getenv("KAFKA_SASL_PASSWORD")

# Eval jobs carry the full trace event (can be multi-MB); raise the consumer
# fetch ceiling above the ~1MB default to match the API/broker sizing.
KAFKA_MAX_FETCH_BYTES = int(os.getenv("KAFKA_MAX_FETCH_BYTES", str(10 * 1024 * 1024)))


def kafka_auth_kwargs() -> dict:
    """aiokafka security kwargs derived from env, shared by consumer + producer.

    PLAINTEXT (default, local docker-compose) → no auth.
    SASL_SSL → SCRAM-SHA-512 username/password over TLS (AWS MSK). MSK broker
    certs chain to Amazon Trust Services (in the default CA bundle), so no CA
    file is needed.
    """
    protocol = (KAFKA_SECURITY_PROTOCOL or "PLAINTEXT").upper()
    if protocol == "SASL_SSL":
        return {
            "security_protocol": "SASL_SSL",
            "sasl_mechanism": KAFKA_SASL_MECHANISM,
            "sasl_plain_username": KAFKA_SASL_USERNAME,
            "sasl_plain_password": KAFKA_SASL_PASSWORD,
            "ssl_context": ssl.create_default_context(),
        }
    return {"security_protocol": "PLAINTEXT"}

KAFKA_EVAL_TOPIC = os.getenv("KAFKA_EVAL_TOPIC")
KAFKA_EVAL_GROUP_ID = os.getenv("KAFKA_EVAL_GROUP_ID")

JUDGE_PROVIDER = os.getenv("EVAL_JUDGE_PROVIDER")
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL")
JUDGE_THRESHOLD = float(os.getenv("EVAL_JUDGE_THRESHOLD"))
JUDGE_CACHE_ENABLED = os.getenv("EVAL_JUDGE_CACHE")
JUDGE_CACHE_TTL = float(os.getenv("EVAL_JUDGE_CACHE_TTL"))
JUDGE_CACHE_MAX = int(os.getenv("EVAL_JUDGE_CACHE_MAX"))


CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT"))
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD")
CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE")
CLICKHOUSE_TRACE_TABLE = os.getenv("CLICKHOUSE_TRACE_TABLE")
CLICKHOUSE_TRACE_COSTS_TABLE = os.getenv("CLICKHOUSE_TRACE_COSTS_TABLE")
CLICKHOUSE_EVALUATIONS_TABLE = os.getenv("CLICKHOUSE_EVALUATIONS_TABLE")
CLICKHOUSE_SECURITY_TABLE    = os.getenv("CLICKHOUSE_SECURITY_TABLE")

KAFKA_SECURITY_REPLY_TOPIC   = os.getenv("KAFKA_SECURITY_REPLY_TOPIC")
KAFKA_PLAYGROUND_REPLY_TOPIC = os.getenv("KAFKA_PLAYGROUND_REPLY_TOPIC")

# Postgres holds the admin-editable LLM-as-Judge prompt overrides. Optional:
# when unset, the worker uses its built-in default prompts (fail-open) and the
# Admin "Judge Prompts" tab simply has no effect on this worker.
POSTGRES_DSN = os.getenv("POSTGRES_DSN")
# How long (seconds) a loaded judge-prompt snapshot is trusted before the next
# eval triggers a refresh from Postgres.
JUDGE_PROMPT_CACHE_TTL = float(os.getenv("EVAL_JUDGE_PROMPT_TTL", "60"))