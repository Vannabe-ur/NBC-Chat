import ollama
import re

MODEL_NAME = "qwen3:4b"
FALLBACK_MODEL = "qwen2.5:1.5b"
ENABLE_THINKING = False
MAX_TOKENS = 300   # cap output length so a misbehaving model can't ramble forever

SYSTEM_PROMPT = """You are an assistant that answers questions ONLY about the National Bank of Cambodia (NBC), using ONLY the reference information provided below.

Rules:
- If the reference information contains the answer, answer clearly and concisely based on it.
- If the reference information does NOT contain a relevant answer, respond exactly: "I don't have information about that in my NBC knowledge base."
- Do not use any outside knowledge, assumptions, or general information about central banks.
- Do not speculate or make up details not present in the reference information.
- Write the answer as a friendly, natural sentence rather than returning a bare word or fragment.
- For short-form or abbreviation questions, use wording such as "It is called NBC." or "The short form is NBC."
- Keep simple factual answers to one concise sentence unless more explanation is needed.
"""


def build_system_prompt(enable_thinking: bool) -> str:
    """Qwen3's own chat template respects /no_think as a hard suffix instruction,
    independent of whether the API-level think flag is honored."""
    suffix = "" if enable_thinking else "\n/no_think"
    return SYSTEM_PROMPT + suffix


def build_context_block(matches: list[dict]) -> str:
    return "\n\n".join(f"Q: {m['question']}\nA: {m['answer']}" for m in matches)


def strip_thinking(text: str) -> str:
    """Remove leaked reasoning, tagged or not."""
    # Case 1: properly tagged <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Case 2: untagged leak that ends with a closing </think> only
    # cut everything up to and including it.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    return text.strip()


def generate_answer(question: str, retrieval_result: dict, model: str = MODEL_NAME,
                     enable_thinking: bool = ENABLE_THINKING) -> str:
    if not retrieval_result["is_confident"]:
        return "I don't have information about that in my NBC knowledge base."

    context = build_context_block(retrieval_result["matches"])
    system_prompt = build_system_prompt(enable_thinking)
    user_prompt = f"""Reference information:
{context}

User question: {question}

Answer using only the reference information above."""

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        think=enable_thinking,
        options={"num_predict": MAX_TOKENS},
    )

    # --- Diagnostics ---
    load_ms = response.get("load_duration", 0) / 1e6
    eval_tokens = response.get("eval_count", 0)
    eval_ms = response.get("eval_duration", 0) / 1e6
    print(f"[timing] load={load_ms:.0f}ms  tokens={eval_tokens}  eval={eval_ms:.0f}ms")

    raw_content = response["message"]["content"]
    final_answer = strip_thinking(raw_content)

    if enable_thinking and response["message"].get("thinking"):
        print("--- Reasoning trace (backend log only) ---")
        print(response["message"]["thinking"])

    return final_answer


if __name__ == "__main__":
    from src.backend.retriever import retrieve

    question = "What is the shot form of National Bank of Cambodia?"
    result = retrieve(question)

    print("=== Non-thinking mode ===")
    print(generate_answer(question, result, enable_thinking=False))

    print("\n=== Thinking mode ===")
    print(generate_answer(question, result, enable_thinking=True))
