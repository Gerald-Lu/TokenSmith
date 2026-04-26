import json
import pickle
import random
from pathlib import Path

import pytest

INDEX_DIR = Path(__file__).resolve().parents[1] / "index" / "sections"
INDEX_PREFIX = "textbook_index"


def _artifact_paths():
    base = INDEX_DIR / INDEX_PREFIX
    return {
        "faiss": INDEX_DIR / f"{INDEX_PREFIX}.faiss",
        "chunks": Path(str(base) + "_chunks.pkl"),
        "meta": Path(str(base) + "_meta.pkl"),
        "parent_map": INDEX_DIR / f"{INDEX_PREFIX}_parent_chunk_map.json",
        "page_map": INDEX_DIR / f"{INDEX_PREFIX}_page_to_chunk_map.json",
    }


@pytest.fixture(scope="module")
def hier_index_bundle():
    p = _artifact_paths()
    need = [p["faiss"], p["chunks"], p["meta"], p["parent_map"]]
    if not all(x.exists() for x in need):
        pytest.skip("hierarchical index artifacts missing under index/sections/")
    import faiss
    faiss_index = faiss.read_index(str(p["faiss"]))
    chunks = pickle.loads(p["chunks"].read_bytes())
    metadata = pickle.loads(p["meta"].read_bytes())
    parent_map = json.loads(p["parent_map"].read_text())
    page_map = json.loads(p["page_map"].read_text()) if p["page_map"].exists() else {}
    return {"faiss": faiss_index, "chunks": chunks, "metadata": metadata, "parent_map": parent_map, "page_map": page_map}


def test_index_lengths_aligned(hier_index_bundle):
    faiss_index = hier_index_bundle["faiss"]
    chunks = hier_index_bundle["chunks"]
    metadata = hier_index_bundle["metadata"]
    parent_map = hier_index_bundle["parent_map"]
    assert len(chunks) == len(metadata) == len(parent_map) == faiss_index.ntotal


def test_metadata_chunk_ids_match_position(hier_index_bundle):
    metadata = hier_index_bundle["metadata"]
    for i, m in enumerate(metadata):
        assert m.get("chunk_id") == i


def test_every_chunk_has_parent_entry(hier_index_bundle):
    chunks = hier_index_bundle["chunks"]
    parent_map = hier_index_bundle["parent_map"]
    for i in range(len(chunks)):
        v = parent_map.get(str(i), parent_map.get(i))
        assert v is not None and isinstance(v, str) and len(v) > 0


def test_parent_contains_child_snippet(hier_index_bundle):
    chunks = hier_index_bundle["chunks"]
    metadata = hier_index_bundle["metadata"]
    parent_map = hier_index_bundle["parent_map"]
    n = len(chunks)
    rng = random.Random(42)
    sample = rng.sample(range(n), min(50, n))
    for i in sample:
        child = chunks[i]
        parent = parent_map.get(str(i)) or parent_map.get(i)
        assert parent is not None
        pv = (metadata[i].get("text_preview") or child).strip()[:80]
        if not pv:
            continue
        assert pv in parent, f"chunk {i}: parent missing child prefix"


def test_no_introduction_chunks_leaked(hier_index_bundle):
    metadata = hier_index_bundle["metadata"]
    for m in metadata:
        assert m.get("section") != "Introduction"


def test_page_to_chunk_map_within_bounds(hier_index_bundle):
    chunks = hier_index_bundle["chunks"]
    page_map = hier_index_bundle["page_map"]
    n = len(chunks)
    for _page, ids in page_map.items():
        for cid in ids:
            assert 0 <= int(cid) < n


def test_faiss_ntotal_matches_corpus(hier_index_bundle):
    assert hier_index_bundle["faiss"].ntotal == len(hier_index_bundle["chunks"])
