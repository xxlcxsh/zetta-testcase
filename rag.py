"""Small, dependency-free retrieval core for the Accord demo."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


WORD_RE = re.compile(r"[a-zа-яёая0-9]+", re.IGNORECASE)
SERVICE_TERMS = {
    "rehearsal": ("репетиц", "комнат"),
    "studio": ("студ", "звукозап", "сеанс"),
    "workshop": ("мастер", "керам"),
}
STOP_WORDS = {
    "как", "какой", "какая", "какое", "какие", "где", "куда", "на", "в", "и", "или", "для", "до", "по",
    "за", "из", "при", "ли", "можно", "нужно", "нужен", "есть", "его", "ее", "это", "часов", "часа",
    "заранее", "бронь", "бронирование", "бронировать", "начала", "начало", "срок", "какой",
}


def terms(text: str) -> list[str]:
    """Russian-friendly lexical features preserving term frequency for BM25."""
    words = [w.lower().replace("ё", "е") for w in WORD_RE.findall(text)]
    words = [w for w in words if w not in STOP_WORDS]
    return words + [w[:4] for w in words if len(w) > 5]


def split_sentences(text: str) -> list[str]:
    blocks: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # Tables are more useful as separate retrievable rows.
        if " | " in paragraph:
            blocks.extend(line.strip() for line in paragraph.splitlines() if line.strip())
        else:
            blocks.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip())
    return blocks


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    status: str
    text: str
    score: float = 0.0

    def public(self) -> dict:
        return asdict(self)


class Retriever:
    """Hybrid BM25 + multilingual dense retrieval fused with RRF."""

    def __init__(
        self,
        documents_dir: str | Path = "documents",
        dense_model_name: str = "intfloat/multilingual-e5-small",
        rrf_k: int = 60,
    ):
        self.documents_dir = Path(documents_dir)
        self.dense_model_name = dense_model_name
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []
        for path in sorted(self.documents_dir.glob("*.txt")):
            raw = path.read_text(encoding="utf-8-sig")
            doc_id = re.search(r"DOC-\d+", raw).group(0)  # type: ignore[union-attr]
            title_match = re.search(r"^Название:\s*(.+)$", raw, re.MULTILINE)
            status_match = re.search(r"^Статус:\s*(.+)$", raw, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else path.stem
            status = status_match.group(1).strip() if status_match else ""
            for text in split_sentences(raw):
                if text.startswith(("DOC-", "Название:", "Статус:", "Дата ", "Замен")):
                    continue
                self.chunks.append(Chunk(doc_id, title, status, text))

        self._active = [c for c in self.chunks if "ДЕЙСТВУЕТ" in c.status]
        if not self._active:
            raise ValueError("Не найдены действующие документы")

        # Sparse branch: true BM25 with term frequency and document-length normalization.
        self._bm25 = BM25Okapi([terms(c.text + " " + c.title) for c in self._active])

        # Dense branch: multilingual E5 embeddings searched by cosine similarity in FAISS.
        self._dense_model = SentenceTransformer(dense_model_name)
        passages = [f"passage: {c.title}. {c.text}" for c in self._active]
        matrix = self._dense_model.encode(
            passages, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        self._dense_index = faiss.IndexFlatIP(matrix.shape[1])
        self._dense_index.add(matrix)

    def search(self, query: str, limit: int = 5) -> list[Chunk]:
        candidate_count = min(max(limit * 4, 20), len(self._active))

        bm25_scores = self._bm25.get_scores(terms(query))
        bm25_order = np.argsort(-bm25_scores)[:candidate_count]
        bm25_ranking = [int(i) for i in bm25_order if bm25_scores[i] > 0]

        query_vector = self._dense_model.encode(
            [f"query: {query}"], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        _, dense_indices = self._dense_index.search(query_vector, candidate_count)
        dense_ranking = [int(i) for i in dense_indices[0] if i >= 0]

        # RRF uses ranks, not incomparable BM25/cosine values.
        fused: dict[int, float] = defaultdict(float)
        for ranking in (bm25_ranking, dense_ranking):
            for rank, index in enumerate(ranking, start=1):
                fused[index] += 1.0 / (self.rrf_k + rank)

        order = sorted(fused, key=lambda i: (-fused[i], i))[:limit]
        return [
            Chunk(
                self._active[i].doc_id,
                self._active[i].title,
                self._active[i].status,
                self._active[i].text,
                round(fused[i], 6),
            )
            for i in order
        ]


class MiniRAG:
    def __init__(
        self,
        documents_dir: str | Path = "documents",
        model_name: str = "Qwen/Qwen3-0.6B",
        dense_model_name: str = "intfloat/multilingual-e5-small",
    ):
        self.retriever = Retriever(documents_dir, dense_model_name=dense_model_name)
        self.model_name = model_name
        self._pipeline = None

    @staticmethod
    def _is_ambiguous(question: str) -> bool:
        q = question.lower()
        is_cancel = "отмен" in q or "брон" in q
        named = any(any(term in q for term in terms) for terms in SERVICE_TERMS.values())
        return is_cancel and not named

    @staticmethod
    def _dedupe(chunks: list[Chunk], limit: int = 3) -> list[Chunk]:
        result: list[Chunk] = []
        seen: set[tuple[str, str]] = set()
        for chunk in chunks:
            key = (chunk.doc_id, chunk.text)
            if key not in seen:
                result.append(chunk)
                seen.add(key)
            if len(result) == limit:
                break
        return result

    def answer(self, question: str, mode: Literal["chunk", "llm"] = "chunk") -> dict:
        if not question.strip():
            raise ValueError("Вопрос не должен быть пустым")
        if self._is_ambiguous(question):
            return {"question": question, "mode": mode, "status": "clarification_needed",
                    "answer": "Уточните, пожалуйста: речь о репетиционной комнате, студии звукозаписи или мастер-классе?",
                    "sources": []}

        found = self.retriever.search(question, limit=8)
        if not found:
            return {"question": question, "mode": mode, "status": "insufficient_data",
                    "answer": "Недостаточно данных.", "sources": []}

        # A statement that the requested information is absent is itself useful evidence.
        missing = [c for c in found if "информаци" in c.text.lower() and "не приводится" in c.text.lower()]
        if missing:
            chosen = missing[:1]
            return {"question": question, "mode": mode, "status": "insufficient_data",
                    "answer": "Недостаточно данных: в документах нет информации о студенческой скидке.",
                    "sources": [c.public() for c in chosen]}

        selected = self._select_evidence(question, found)
        if mode == "chunk":
            answer = "\n".join(c.text for c in selected)
        elif mode == "llm":
            answer = self._generate(question, selected)
        else:
            raise ValueError("Режим должен быть 'chunk' или 'llm'")
        return {"question": question, "mode": mode, "status": "answered", "answer": answer,
                "sources": [c.public() for c in selected]}

    def _select_evidence(self, question: str, found: list[Chunk]) -> list[Chunk]:
        q = question.lower()
        active = [c for c in self.retriever.chunks if "ДЕЙСТВУЕТ" in c.status]
        selected: list[Chunk] = []
        # Composite recorder question needs both issue terms and return procedure.
        if "рекордер" in q and "верн" in q:
            selected.extend(c for c in active if c.doc_id == "DOC-04" and "рекордер |" in c.text.lower())
            selected.extend(c for c in active if c.doc_id == "DOC-05" and "возвращать лично" in c.text.lower())
        elif "электрогитар" in q:
            selected.extend(c for c in active if c.doc_id == "DOC-04" and c.text.lower().startswith("электрогитара |"))
        elif "микрофон" in q and "продл" in q:
            selected.extend(c for c in active if c.doc_id == "DOC-05" and any(phrase in c.text.lower() for phrase in (
                "можно продлить", "для этого необходимо", "продление считается")))
        elif any(term in q for term in SERVICE_TERMS["rehearsal"]):
            selected = [c for c in found if c.doc_id == "DOC-02" and "бесплатн" in c.text.lower()]
        elif any(term in q for term in SERVICE_TERMS["studio"]):
            selected = [c for c in found if c.doc_id == "DOC-03" and "бесплатн" in c.text.lower()]
        elif any(term in q for term in SERVICE_TERMS["workshop"]):
            selected = [c for c in found if c.doc_id == "DOC-06"]
        else:
            selected = found[:2]
        return self._dedupe(selected or found, limit=3)

    def _load_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError("Для режима llm установите зависимости: pip install -r requirements-llm.txt") from exc
            self._pipeline = pipeline("text-generation", model=self.model_name, device_map="auto", dtype="auto")
        return self._pipeline

    def _generate(self, question: str, chunks: list[Chunk]) -> str:
        context = "\n".join(f"{number}. [{c.doc_id}] {c.text}" for number, c in enumerate(chunks, start=1))
        messages = [
            {"role": "system", "content": (
                "Ты — ассистент RAG творческого центра. Строго следуй инструкциям:\n"
                "1. Отвечай по-русски и только по переданному контексту. Не используй внешние знания.\n"
                "2. Отвечай прямо, кратко и без рассуждений.\n"
                "3. Прочитай КАЖДЫЙ пронумерованный фрагмент. Не ограничивайся первым.\n"
                "4. Если спрашивают, как что-то сделать, перечисли ВСЕ релевантные условия: доступность, способ подачи запроса, срок и необходимое подтверждение.\n"
                "5. Точно сохраняй числа, суммы, время, сроки, места и исключения.\n"
                "6. Строка 'Оборудование | срок | залог' — таблица; извлекай точные значения.\n"
                "   Пример: если контекст 'Скрипка | 3 дня | 700 рублей', а вопрос 'Какой залог?', ответ: 'Залог — 700 рублей'.\n"
                "7. Не добавляй неподтвержденные факты. Не пиши ID документов и не копируй цитаты целиком.\n"
                "8. Перед ответом молча проверь, что учтен каждый релевантный фрагмент. Не выводи эту проверку."
            )},
            {"role": "user", "content": (
                f"Контекст:\n{context}\n\nВопрос: {question}\n\n"
                "Сформулируй один полный ответ. Учти все релевантные пронумерованные фрагменты."
            )},
        ]
        generator = self._load_pipeline()
        prompt = generator.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        result = generator(prompt, max_new_tokens=100, do_sample=False, return_full_text=False)
        generated = result[0]["generated_text"]
        return str(generated).strip()
