from typing import Any, Dict, List, Optional, Sequence
from tests.metrics.base import MetricBase

def _per_chunk_pages(chunks_info, metadata, chunks, parent_map):
    if not chunks_info or not metadata:
        return []

    lookup = {}
    if chunks is not None and any(c.get("chunk_id") == "reranked_chunk" for c in chunks_info):
        for cid, text in enumerate(chunks):
            lookup.setdefault(text, cid)
        if parent_map:
            for k, ptext in parent_map.items():
                try:
                    cid = int(k)
                except (TypeError, ValueError):
                    continue
                lookup.setdefault(ptext, cid)

    out = []
    for c in chunks_info:
        cid = c.get("chunk_id")
        if cid == "reranked_chunk" or not isinstance(cid, int):
            content = c.get("content")
            if isinstance(content, list):
                content = content[0] if content else ""
            cid = lookup.get(content)
            if cid is None:
                out.append([])
                continue
        if 0 <= cid < len(metadata):
            out.append([int(p) for p in (metadata[cid].get("page_numbers", []) or [])])
        else:
            out.append([])
    return out

class PageRecallMetric(MetricBase):
    def __init__(self, k: int = 5):
        self.k = k

    @property
    def name(self) -> str:
        return "page_recall"

    @property
    def weight(self) -> float:
        return 0.0

    def calculate(
        self,
        gold_pages: Optional[Sequence[int]],
        chunks_info: Optional[Sequence[Dict[str, Any]]],
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
        chunks: Optional[Sequence[str]] = None,
        parent_map: Optional[Dict[Any, str]] = None,
    ) -> float:
        if not gold_pages:
            return 0.0
        gold = {int(p) for p in gold_pages}
        per_chunk = _per_chunk_pages(chunks_info or [], metadata, chunks, parent_map)
        retrieved = {p for pgs in per_chunk[: self.k] for p in pgs}
        if not retrieved:
            return 0.0
        return len(gold & retrieved) / len(gold)
