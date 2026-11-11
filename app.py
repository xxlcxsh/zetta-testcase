from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rag import MiniRAG


BASE_DIR = Path(__file__).resolve().parent
rag = MiniRAG(BASE_DIR / "documents")
app = FastAPI(title="Аккорд: mini-RAG", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    mode: str = Field(default="chunk", pattern="^(chunk|llm)$")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "generator": rag.model_name,
        "retrieval": "BM25 + multilingual-e5-small/FAISS + RRF(k=60)",
    }


@app.post("/api/ask")
def ask(payload: AskRequest) -> dict:
    try:
        return rag.answer(payload.question, payload.mode)  # type: ignore[arg-type]
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503 if isinstance(exc, RuntimeError) else 400, detail=str(exc)) from exc
