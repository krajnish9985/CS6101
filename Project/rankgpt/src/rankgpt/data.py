"""Loading the master reranking file produced by Member 1.

Expected JSONL schema (one object per query):
    {
      "query_id": str,
      "query_text": str,
      "retrieved_docs": [
          {"rank": int, "doc_id": str, "bm25_score": float,
           "title": str, "text": str, "relevance": int}, ...
      ],
      "relevant_docs": [...],
      "stage1_stats": {...}
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass
class Candidate:
    """One BM25 candidate passage for a query."""

    doc_id: str
    title: str
    text: str
    bm25_score: float
    bm25_rank: int
    relevance: int = 0

    def content(self, max_words: Optional[int] = None, include_title: bool = True) -> str:
        """Passage text as shown to the LLM."""
        parts = []
        if include_title and self.title:
            parts.append(self.title.strip())
        if self.text:
            parts.append(self.text.strip())
        body = " ".join(parts)
        body = " ".join(body.split())  # collapse whitespace
        if max_words is not None and max_words > 0:
            words = body.split()
            if len(words) > max_words:
                body = " ".join(words[:max_words])
        return body


@dataclass
class QueryRecord:
    """A query with its ordered BM25 candidate list."""

    query_id: str
    query_text: str
    candidates: List[Candidate] = field(default_factory=list)

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)


def iter_reranking_data(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_reranking_data(
    path,
    top_k: Optional[int] = None,
    limit: Optional[int] = None,
    query_ids: Optional[List[str]] = None,
) -> List[QueryRecord]:
    """Load queries and their BM25 candidates.

    Args:
        path: path to reranking_data.jsonl
        top_k: keep only the first `top_k` BM25 candidates per query
        limit: keep only the first `limit` queries (useful for smoke tests)
        query_ids: if given, keep only these query ids
    """
    wanted = set(query_ids) if query_ids else None
    records: List[QueryRecord] = []

    for obj in iter_reranking_data(path):
        qid = str(obj["query_id"])
        if wanted is not None and qid not in wanted:
            continue

        docs = obj.get("retrieved_docs", [])
        # Defensive: the file should already be rank-ordered, but do not rely on it.
        docs = sorted(docs, key=lambda d: int(d["rank"]))
        if top_k is not None:
            docs = docs[:top_k]

        candidates = [
            Candidate(
                doc_id=str(d["doc_id"]),
                title=d.get("title") or "",
                text=d.get("text") or "",
                bm25_score=float(d.get("bm25_score", 0.0)),
                bm25_rank=int(d["rank"]),
                relevance=int(d.get("relevance", 0)),
            )
            for d in docs
        ]

        if not candidates:
            # Queries with zero BM25 hits exist in some BEIR datasets (e.g. nfcorpus).
            continue

        records.append(
            QueryRecord(query_id=qid, query_text=obj["query_text"], candidates=candidates)
        )

        if limit is not None and len(records) >= limit:
            break

    return records


def load_full_candidate_lists(path) -> dict:
    """query_id -> full ordered list of doc_ids (all 100), for run-file padding."""
    out = {}
    for obj in iter_reranking_data(path):
        docs = sorted(obj.get("retrieved_docs", []), key=lambda d: int(d["rank"]))
        out[str(obj["query_id"])] = [str(d["doc_id"]) for d in docs]
    return out
