"""Retrieval core: chunking, TF-IDF embeddings, cosine search, answer building.

Embeddings are computed locally (no API key needed). The Embedder interface is
small so a neural embedding model can be swapped in without touching the API.
"""
import math
import re
from collections import Counter

import numpy as np

STOP = set("a an the and or of to in on for with is are was were be by as at it this that from".split())
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def chunk_text(text: str, size: int = 400, overlap: int = 60) -> list[str]:
    """Split on sentence boundaries into ~size-char chunks with overlap."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > size:
            chunks.append(cur)
            cur = cur[-overlap:].split(" ", 1)[-1] + " " + s if overlap else s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks


class TfidfEmbedder:
    """Fits a vocabulary + IDF on a corpus, embeds text as L2-normalised vectors."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        df = Counter()
        for t in texts:
            df.update(set(tokenize(t)))
        self.vocab = {w: i for i, w in enumerate(sorted(df))}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[w])) + 1 for w in self.vocab])
        return self

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for w, c in Counter(tokenize(text)).items():
            i = self.vocab.get(w)
            if i is not None:
                v[i] = (1 + math.log(c)) * self.idf[i]
        n = np.linalg.norm(v)
        return v / n if n else v


class Index:
    """In-memory vector index over (doc_id, doc_name, text) chunks."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.embedder = TfidfEmbedder().fit([c["text"] for c in chunks])
        self.matrix = (
            np.vstack([self.embedder.embed(c["text"]) for c in chunks]) if chunks else np.zeros((0, 0))
        )

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> list[dict]:
        if not self.chunks:
            return []
        q = self.embedder.embed(query)
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        return [
            {**self.chunks[i], "score": round(float(scores[i]), 4)}
            for i in order
            if scores[i] >= min_score
        ]


def build_prompt(question: str, hits: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] ({h['doc_name']}) {h['text']}" for i, h in enumerate(hits))
    return (
        "Answer the question using ONLY the numbered context. Cite sources like [1]. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}"
    )


def extractive_answer(question: str, hits: list[dict]) -> str:
    """Offline fallback: return the best-matching sentence from the top chunk, cited."""
    if not hits:
        return "I don't know: no relevant passage found in the uploaded documents."
    q = set(tokenize(question))
    best, best_overlap = hits[0]["text"], -1
    for s in re.split(r"(?<=[.!?])\s+", hits[0]["text"]):
        o = len(q & set(tokenize(s)))
        if o > best_overlap:
            best, best_overlap = s, o
    return f"{best} [1]"


def generate_answer(question: str, hits: list[dict], llm=None) -> tuple[str, str]:
    """Returns (answer, mode). Refuses to answer without retrieved context."""
    if not hits:
        return "I don't know: no relevant passage found in the uploaded documents.", "no-context"
    if llm is not None:
        return llm(build_prompt(question, hits)), "llm"
    return extractive_answer(question, hits), "extractive"
