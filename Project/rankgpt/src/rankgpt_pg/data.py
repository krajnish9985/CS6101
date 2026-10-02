"""Loading of Member 1's BM25 candidate file.

One record per query, as produced by ``build_reranking_dataset``:

    {"query_id": str,
     "query_text": str,
     "retrieved_docs": [{"rank", "doc_id", "bm25_score",
                         "title", "text", "relevance"}, ...],
     "relevant_docs": [...],
     "stage1_stats": {...}}

``retrieved_docs`` is already in BM25 rank order, so slicing the first
``depth`` entries gives the top-k candidate set with no sorting needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

Record = Dict[str, Any]
Doc = Dict[str, Any]

REQUIRED_RECORD_KEYS = ("query_id", "query_text", "retrieved_docs")
REQUIRED_DOC_KEYS = ("rank", "doc_id", "title", "text", "relevance")


def iter_records(path: str | Path) -> Iterator[Record]:
    """Yield one validated record per non-empty line of the JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"candidate file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} is not valid JSON: {error.msg}"
                ) from error

            missing = [key for key in REQUIRED_RECORD_KEYS if key not in record]
            if missing:
                raise ValueError(
                    f"{path}:{line_number} missing key(s): {missing}"
                )

            yield record


def load_records(path: str | Path, limit: int | None = None) -> List[Record]:
    """Read the whole file into memory; SciFact is 300 queries, so this is fine.

    ``limit`` reads only the first N queries, which is how we scale a run
    up from 1 to 5 to 50 to all of them.
    """
    records: List[Record] = []
    for record in iter_records(path):
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def find_record(records: List[Record], query_id: str) -> Record:
    """Look up one record by query_id, for targeted debugging."""
    for record in records:
        if str(record["query_id"]) == str(query_id):
            return record
    raise KeyError(f"query_id {query_id!r} not found among {len(records)} records")


def take_candidates(record: Record, depth: int) -> List[Doc]:
    """Return the top-``depth`` BM25 candidates for one query.

    A few BEIR queries retrieve fewer than ``depth`` documents, so the
    caller must handle short lists rather than assume a fixed width.
    """
    docs = record["retrieved_docs"][:depth]

    for position, doc in enumerate(docs):
        missing = [key for key in REQUIRED_DOC_KEYS if key not in doc]
        if missing:
            raise ValueError(
                f"query {record['query_id']} candidate at position {position} "
                f"missing key(s): {missing}"
            )

    return docs


def gold_doc_ids(record: Record) -> set[str]:
    """Positively judged doc_ids for this query, from Member 1's qrels join."""
    return {
        str(doc["doc_id"])
        for doc in record.get("relevant_docs", [])
        if int(doc.get("relevance", 0)) > 0
    }


def gold_grades(record: Record) -> List[int]:
    """Every positive relevance grade judged for this query, from qrels.

    Used as the ideal pool for standard BEIR nDCG, so that documents
    stage-1 retrieval missed still count against the score.
    """
    return [
        int(doc.get("relevance", 0))
        for doc in record.get("relevant_docs", [])
        if int(doc.get("relevance", 0)) > 0
    ]

def describe_dataset(records: List[Record], depth: int) -> Dict[str, Any]:
    """Summary stats, printed at the start of a run so the log is self-documenting."""
    depths = [len(record["retrieved_docs"]) for record in records]
    n_with_gold_in_window = sum(
        1
        for record in records
        if any(
            int(doc.get("relevance", 0)) > 0
            for doc in record["retrieved_docs"][:depth]
        )
    )

    return {
        "n_queries": len(records),
        "min_candidates": min(depths) if depths else 0,
        "max_candidates": max(depths) if depths else 0,
        "n_short_of_depth": sum(1 for value in depths if value < depth),
        "n_queries_with_gold_in_top_k": n_with_gold_in_window,
        "depth": depth,
    }