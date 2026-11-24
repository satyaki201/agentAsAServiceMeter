"""Usage metering, pricing and storage.

Knows nothing about the agent: it only needs a user, a session, and the
token counts / latencies the caller reports. Reusable for any LLM service.
"""
import sqlite3
import time
import uuid
from decimal import Decimal

DB_PATH = "usage.db"

# =====================================================================
# Pricing (kept in config, never hard-coded in the agent loop)
# Prices are INR per 1M tokens. Because 1 INR = 1,000,000 micro-INR,
# "INR per 1M tokens" is exactly "micro-INR per token", so cost math is
# plain multiplication with no float rounding drift.
# ---------------------------------------------------------------------
PRICE_VERSION = "2026-11-v1"   # stored on every event; bump when prices change
PRICING = {
    "qwen3:8b-q4_K_M": {
        "input_per_1m": Decimal("20"),    # INR per 1M prompt tokens
        "output_per_1m": Decimal("80"),   # INR per 1M completion tokens
    },
}
TOOL_CALL_FEE_MICRO = {"calculator": 0}   # per-call fee in micro-INR (0 = free)


def estimate_tokens(text):
    """Rough estimate (~4 chars/token). Only used for approximations."""
    return max(1, len(text) // 4)


def llm_cost_micro(model, prompt_tokens, completion_tokens):
    p = PRICING[model]
    cost = (Decimal(prompt_tokens) * p["input_per_1m"]
            + Decimal(completion_tokens) * p["output_per_1m"])
    return int(cost.to_integral_value())


def micro_to_inr(micro):
    return f"₹{micro / 1_000_000:.6f}"


# =====================================================================
# Storage
# =====================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    event_id               TEXT PRIMARY KEY,   -- unique -> idempotent inserts
    ts                     REAL NOT NULL,
    user_id                TEXT NOT NULL,
    session_id             TEXT NOT NULL,
    request_id             TEXT NOT NULL,
    step                   INTEGER NOT NULL,
    kind                   TEXT NOT NULL,      -- 'llm' | 'tool'
    name                   TEXT NOT NULL,      -- model or tool name
    status                 TEXT NOT NULL,      -- 'ok' | 'ok_estimated' | 'error'
    prompt_tokens          INTEGER NOT NULL DEFAULT 0,
    completion_tokens      INTEGER NOT NULL DEFAULT 0,
    system_overhead_tokens INTEGER NOT NULL DEFAULT 0,  -- approx, subset of prompt_tokens
    latency_ms             INTEGER NOT NULL DEFAULT 0,
    cost_micro             INTEGER NOT NULL DEFAULT 0,  -- micro-INR; 0 for errors
    price_version          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_user    ON usage_events (user_id, ts);
CREATE INDEX IF NOT EXISTS idx_usage_session ON usage_events (session_id);
"""


# =====================================================================
# Meter
# =====================================================================
class Meter:
    """Tracks usage for one (user, session). Every call is logged to SQLite.

    system_overhead_tokens: approximate fixed tokens (system prompt + tool
    schema) sent on every LLM call. Stored per event for later billing policy;
    it is NOT billed separately.
    """

    def __init__(self, user_id, session_id=None, token_budget=None,
                 system_overhead_tokens=0, db_path=DB_PATH):
        self.user_id = user_id
        self.session_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        self.token_budget = token_budget
        self.system_overhead_tokens = system_overhead_tokens
        self.request_id = None
        self.step = 0

        # in-memory running totals for this session (fast budget checks)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_calls = 0
        self.tool_calls = 0
        self.errors = 0
        self.cost_micro = 0

        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---- request lifecycle ----
    def new_request(self):
        self.request_id = f"req_{uuid.uuid4().hex[:12]}"
        self.step = 0
        return self.request_id

    # ---- recording ----
    def _insert(self, **f):
        self.conn.execute(
            """INSERT OR IGNORE INTO usage_events
               (event_id, ts, user_id, session_id, request_id, step, kind, name,
                status, prompt_tokens, completion_tokens, system_overhead_tokens,
                latency_ms, cost_micro, price_version)
               VALUES (:event_id, :ts, :user_id, :session_id, :request_id, :step,
                       :kind, :name, :status, :prompt_tokens, :completion_tokens,
                       :system_overhead_tokens, :latency_ms, :cost_micro,
                       :price_version)""",
            {
                "event_id": uuid.uuid4().hex,
                "ts": time.time(),
                "user_id": self.user_id,
                "session_id": self.session_id,
                "request_id": self.request_id,
                "step": self.step,
                "price_version": PRICE_VERSION,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "system_overhead_tokens": 0,
                "latency_ms": 0,
                "cost_micro": 0,
                **f,
            },
        )
        self.conn.commit()

    def record_llm(self, model, prompt_tokens, completion_tokens, latency_ms,
                   status="ok"):
        self.step += 1
        # failed calls are logged but never billed
        cost = 0 if status == "error" else llm_cost_micro(model, prompt_tokens, completion_tokens)
        self._insert(
            kind="llm", name=model, status=status,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            system_overhead_tokens=self.system_overhead_tokens if status != "error" else 0,
            latency_ms=latency_ms, cost_micro=cost,
        )
        self.llm_calls += 1
        self.errors += status == "error"
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_micro += cost

    def record_llm_usage(self, model, usage, latency_ms, prompt_text="", completion_text=""):
        """Record from an OpenAI-style `response.usage`.

        If the server reported no usage, fall back to an estimate from the
        given texts and flag the event as 'ok_estimated'.
        """
        if usage is not None:
            self.record_llm(model, usage.prompt_tokens, usage.completion_tokens, latency_ms)
        else:
            self.record_llm(model, estimate_tokens(prompt_text),
                            estimate_tokens(completion_text), latency_ms,
                            status="ok_estimated")

    def record_llm_error(self, model, latency_ms):
        self.record_llm(model, 0, 0, latency_ms, status="error")

    def record_tool(self, name, latency_ms, status="ok"):
        self.step += 1
        cost = 0 if status == "error" else TOOL_CALL_FEE_MICRO.get(name, 0)
        self._insert(kind="tool", name=name, status=status,
                     latency_ms=latency_ms, cost_micro=cost)
        self.tool_calls += 1
        self.errors += status == "error"
        self.cost_micro += cost

    # ---- budget / reporting ----
    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    def over_budget(self):
        return self.token_budget is not None and self.total_tokens >= self.token_budget

    def summary(self):
        return (
            f"session={self.session_id} user={self.user_id} | "
            f"llm_calls={self.llm_calls} tool_calls={self.tool_calls} errors={self.errors} | "
            f"in={self.prompt_tokens} out={self.completion_tokens} "
            f"(budget {self.total_tokens}/{self.token_budget or '∞'}) | "
            f"cost={micro_to_inr(self.cost_micro)}"
        )


def usage_report(user_id, db_path=DB_PATH):
    """Aggregate a user's lifetime usage straight from the event log."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """SELECT COUNT(DISTINCT session_id),
                  SUM(kind='llm'), SUM(kind='tool'),
                  COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0),
                  COALESCE(SUM(cost_micro),0)
           FROM usage_events WHERE user_id = ?""",
        (user_id,),
    ).fetchone()
    conn.close()
    sessions, llm, tools, p, c, cost = row
    return (
        f"user={user_id} | sessions={sessions} llm_calls={llm or 0} "
        f"tool_calls={tools or 0} | in={p} out={c} | total={micro_to_inr(cost)}"
    )