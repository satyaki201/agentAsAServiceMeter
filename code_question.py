"""The math agent. Metering is delegated entirely to metering.Meter."""
import json
import time

from openai import OpenAI

from metering import Meter, estimate_tokens, usage_report

MODEL = "qwen3:8b-q4_K_M"
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

DEMO_USERS = {
    "user_demo_001": {"name": "Demo User", "session_token_budget": 20_000},
}
DEMO_USER_ID = "user_demo_001"


# ---------- The tool itself ----------
def calculator(a, b, operation):
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else "Error: Division by zero"
    elif operation == "power":
        return a ** b
    elif operation == "modulo":
        return a % b if b != 0 else "Error: Division by zero"
    return "Error: Invalid operation"


TOOL_REGISTRY = {"calculator": calculator}

# ---------- Tool description: this is what the model reasons over ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Performs ONE exact arithmetic operation on two numbers. "
                "Use it when a numeric result is needed and mental arithmetic "
                "could be error-prone (large numbers, decimals, many digits), "
                "including intermediate numeric steps inside bigger problems "
                "(e.g. evaluating a derivative or integral at a specific value). "
                "Do NOT use it for explaining concepts, definitions, symbolic "
                "reasoning, or trivial arithmetic you can do reliably yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First operand"},
                    "b": {"type": "number", "description": "Second operand"},
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide", "power", "modulo"],
                    },
                },
                "required": ["a", "b", "operation"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are a helpful mathematical assistant.

You have a calculator tool. YOU decide whether to use it, based on the question:
- Conceptual or theory questions (what is a limit, explain integration by parts):
  answer directly. Never mention the calculator.
- Calculus/algebra problems: do the symbolic reasoning yourself. Only call the
  calculator for numeric steps where an exact result matters.
- Simple arithmetic you can do reliably: answer directly.
- Big or messy arithmetic: call the calculator (one operation per call, chain
  calls if needed), then use the returned values in your final answer.
Never invent a tool result. Give a clear final answer when done."""

# Approximate fixed tokens sent on every call (system prompt + tool schema)
SYSTEM_OVERHEAD_EST = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(json.dumps(TOOLS))


def make_meter(user_id, session_id=None):
    """Build a Meter for a user, using their budget from the user store."""
    user = DEMO_USERS[user_id]
    return Meter(
        user_id,
        session_id=session_id,
        token_budget=user["session_token_budget"],
        system_overhead_tokens=SYSTEM_OVERHEAD_EST,
    )


def run_agent(user_input, meter, history=None, max_steps=8):
    messages = history or [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_input})
    meter.new_request()

    for _ in range(max_steps):
        # Checked before each LLM call, while `messages` is consistent
        # (no dangling tool calls).
        if meter.over_budget():
            text = "Session token budget reached. Please start a new session or top up."
            messages.append({"role": "assistant", "content": text})
            return text, messages

        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",  # model decides: tool call or plain answer
            )
        except Exception as e:
            meter.record_llm_error(MODEL, int((time.time() - t0) * 1000))
            text = f"Error calling model: {e}"
            messages.append({"role": "assistant", "content": text})
            return text, messages
        latency_ms = int((time.time() - t0) * 1000)

        msg = response.choices[0].message
        meter.record_llm_usage(
            MODEL, getattr(response, "usage", None), latency_ms,
            prompt_text=json.dumps(messages), completion_text=msg.content or "",
        )

        # No tool requested -> model chose to answer directly
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content, messages

        # Model requested tool(s): record that, run them, feed results back
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            name = tc.function.name
            t0 = time.time()
            status = "ok"
            try:
                args = json.loads(tc.function.arguments)
                result = TOOL_REGISTRY[name](**args)
            except Exception as e:
                result = f"Error: {e}"
                status = "error"
            meter.record_tool(name, int((time.time() - t0) * 1000), status=status)
            print(f"  [tool] {name}({tc.function.arguments}) -> {result}")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
            )

    return "Stopped: too many tool steps.", messages


if __name__ == "__main__":
    meter = make_meter(DEMO_USER_ID)
    history = None
    print(f"Math agent ready for {DEMO_USERS[DEMO_USER_ID]['name']} ({DEMO_USER_ID}).")
    print("Commands: 'usage' = session + lifetime usage, 'exit' = quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if q.lower() == "usage":
            print(meter.summary())
            print(usage_report(DEMO_USER_ID))
            continue
        answer, history = run_agent(q, meter, history)
        print(f"\nAgent: {answer}")
        print(f"[meter] {meter.summary()}")