from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama
from time import perf_counter

from fastapi import HTTPException
from .retriever import get_collection
from datetime import datetime, timezone


from .retriever import retrieve
from .llm import (
    build_context_block,
    build_system_prompt,
    strip_thinking,
    MODEL_NAME,
    FALLBACK_MODEL,
)

app = FastAPI(title="NBC RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    enable_thinking: bool = False
    top_k: int = 4


class Source(BaseModel):
    id: str
    question: str
    category: str


class ChatResponse(BaseModel):
    answer: str
    thinking: str | None = None
    used_thinking: bool
    is_confident: bool
    model_used: str
    sources: list[Source]
    runtime_ms: int


def call_model(question: str, context: str, think: bool, model: str):
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": build_system_prompt(think)},
            {"role": "user", "content": f"Reference information:\n{context}\n\nUser question: {question}\n\nAnswer using only the reference information above."},
        ],
        think=think,
    )
    return response

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "nbc-rag-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/records/{record_id}")
def get_record(record_id: int):
    collection = get_collection()

    result = collection.get(
        ids=[str(record_id)],
        include=["documents", "metadatas"],
    )

    if not result["ids"]:
        raise HTTPException(status_code=404, detail="Record not found")

    metadata = result["metadatas"][0]

    return {
        "id": result["ids"][0],
        "question": result["documents"][0],
        "answer": metadata["answer"],
        "category": metadata["category"],
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    started_at = perf_counter()
    retrieval = retrieve(req.question, top_k=req.top_k)

    if not retrieval["is_confident"]:
        return ChatResponse(
            answer="I don't have information about that in my NBC knowledge base.",
            used_thinking=False,
            is_confident=False,
            model_used="none",
            sources=[],
            runtime_ms=round((perf_counter() - started_at) * 1000),
        )

    context = build_context_block(retrieval["matches"])
    model_used = MODEL_NAME

    try:
        response = call_model(req.question, context, req.enable_thinking, MODEL_NAME)
    except Exception as e:
        # Primary model failed (timeout, not pulled, crashed) — fall back
        print(f"[WARN] {MODEL_NAME} failed ({e}), falling back to {FALLBACK_MODEL}")
        model_used = FALLBACK_MODEL
        response = call_model(req.question, context, req.enable_thinking, FALLBACK_MODEL)

    raw_content = response["message"]["content"]
    answer = strip_thinking(raw_content)

    # Thinking is used internally by the model, but is never exposed through
    # the API response. Keep it available only in the backend logs.
    if req.enable_thinking:
        thinking_trace = response["message"].get("thinking")
        if thinking_trace:
            print("--- Internal reasoning trace (not returned by API) ---")
            print(thinking_trace)

    sources = [
        Source(id=m["id"], question=m["question"], category=m["category"])
        for m in retrieval["matches"]
    ]

    return ChatResponse(
        answer=answer,
        thinking=None,
        used_thinking=req.enable_thinking,
        is_confident=True,
        model_used=model_used,
        sources=sources,
        runtime_ms=round((perf_counter() - started_at) * 1000),
    )
