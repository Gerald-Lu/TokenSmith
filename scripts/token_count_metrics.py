from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from src.generator import format_prompt, get_llama_model


def _is_numeric_like(value: object) -> bool:
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def _normalize_chunk_content(content: object) -> str:
    if isinstance(content, list):
        if len(content) == 2 and isinstance(content[0], str) and _is_numeric_like(content[1]):
            return content[0]
        return "\n".join(str(c) for c in content)
    return str(content)


def _normalized_chunks_with_stats(row: dict) -> tuple[list[str], dict[str, int]]:
    chunks = []
    stats = {"raw_chunk_count": 0,
        "normalized_chunk_count": 0,
        "tuple_like_chunks": 0,
        "list_chunks": 0,
    }
    raw_chunks = row.get("chunks_info", [])
    stats["raw_chunk_count"] = len(raw_chunks)

    for chunk in raw_chunks:
        content = chunk.get("content", "")
        if isinstance(content, list):
            stats["list_chunks"] += 1
            if len(content) == 2 and isinstance(content[0], str) and _is_numeric_like(content[1]):
                stats["tuple_like_chunks"] += 1
        chunks.append(_normalize_chunk_content(content))

    stats["normalized_chunk_count"] = len(chunks)
    return chunks, stats


def _chunk_preview(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def _collect_chunk_details(rows: list[dict], preview_chars: int) -> list[dict[str, object]]:
    details: list[dict[str, object]] = []
    for row in rows:
        chunks_info = row.get("chunks_info", [])
        row_chunks: list[dict[str, object]] = []
        for idx, chunk in enumerate(chunks_info, start=1):
            content = _normalize_chunk_content(chunk.get("content", ""))
            row_chunks.append(
                {
                    "position": idx,
                    "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                    "source": chunk.get("source", "unknown"),
                    "chars": len(content),
                    "preview": _chunk_preview(content, preview_chars),
                }
            )

        details.append(
            {
                "test_id": row.get("test_id", "unknown"),
                "question": row.get("question", ""),
                "chunk_count": len(row_chunks),
                "chunks": row_chunks,
            }
        )
    return details

def load_results(results_path: Path) -> list[dict]:
    rows = []
    with results_path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
def count_prompt_tokens(
    rows: list[dict],
    verbose: bool = False,
    validate_usage: bool = False,
    n_ctx: int = 4096,
) -> tuple[list[int], list[int], dict[str, int]]:
    if not rows:
        return [], [], {"checked": 0, "matched": 0, "mismatched": 0, "skipped": 0}
    model_path = rows[0]["config"]["model_path"]
    llm = get_llama_model(model_path)
    token_counts: list[int] = []
    prompt_chars: list[int] = []
    usage_validation = {"checked": 0, "matched": 0, "mismatched": 0, "skipped": 0}

    for i, row in enumerate(rows, start=1):
        chunks, chunk_stats = _normalized_chunks_with_stats(row)
            
        prompt = format_prompt(
            chunks,
            row["question"],
            system_prompt_mode=row["config"].get("system_prompt_mode", "baseline"),
        )
        prompt_char_count = len(prompt)
        token_count = len(llm.tokenize(prompt.encode("utf-8"), special=True))
        prompt_chars.append(prompt_char_count)
        token_counts.append(token_count)

        if verbose:
            print(f"[{i}] {row.get('test_id', 'unknown')}")
            print(
                "    chunk_stats: "
                f"raw={chunk_stats['raw_chunk_count']}, "
                f"normalized={chunk_stats['normalized_chunk_count']}, "
                f"list={chunk_stats['list_chunks']}, "
                f"tuple_like={chunk_stats['tuple_like_chunks']}"
            )
            print(f"    prompt_chars: {prompt_char_count}")
            print(f"    prompt_tokens(tokenize,special=True): {token_count}")
            print(f"    chars_per_token: {round(prompt_char_count / token_count, 3) if token_count else 0.0}")

        if validate_usage:
            # Avoid runtime errors when prompt tokens would exceed context window
            if token_count >= n_ctx:
                usage_validation["skipped"] += 1
                if verbose:
                    print(
                        "    usage_prompt_tokens: SKIPPED "
                        f"(token_count={token_count} >= n_ctx={n_ctx})"
                    )
                continue

            usage_validation["checked"] += 1
            usage_prompt_tokens = llm.create_completion(
                prompt,
                max_tokens=1,
                temperature=0.0,
            ).get("usage", {}).get("prompt_tokens")

            if usage_prompt_tokens == token_count:
                usage_validation["matched"] += 1
                if verbose:
                    print(f"    usage_prompt_tokens: {usage_prompt_tokens} (MATCH)")
            else:
                usage_validation["mismatched"] += 1
                if verbose:
                    print(
                        "    usage_prompt_tokens: "
                        f"{usage_prompt_tokens} (MISMATCH vs {token_count})"
                    )

    return token_counts, prompt_chars, usage_validation
def summarize(rows: list[dict]) -> dict[str, object]:
    final_scores = [row["scores"]["final_score"] for row in rows]
    semantic_scores = [row["scores"].get("semantic_similarity", 0.0) for row in rows]
    keyword_scores = [row["scores"].get("keyword_similarity", 0.0) for row in rows]
    nli_scores = [row["scores"].get("nli_similarity", 0.0) for row in rows]
    token_counts, prompt_chars, _ = count_prompt_tokens(rows)
    passed = sum(1 for row in rows if row.get("passed"))
    total = len(rows)
    chars_per_token = [chars / tokens for chars, tokens in zip(prompt_chars, token_counts) if tokens]

    return {"total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "avg_final": round(mean(final_scores), 4) if final_scores else 0.0,
        "avg_semantic": round(mean(semantic_scores), 4) if semantic_scores else 0.0,
        "avg_keyword": round(mean(keyword_scores), 4) if keyword_scores else 0.0,
        "avg_nli": round(mean(nli_scores), 4) if nli_scores else 0.0,
        "prompt_chars_avg": round(mean(prompt_chars), 1) if prompt_chars else 0.0,
        "prompt_chars_min": min(prompt_chars) if prompt_chars else 0,
        "prompt_chars_max": max(prompt_chars) if prompt_chars else 0,
        "prompt_tokens_avg": round(mean(token_counts), 1) if token_counts else 0.0,
        "prompt_tokens_min": min(token_counts) if token_counts else 0,
        "prompt_tokens_max": max(token_counts) if token_counts else 0,
        "chars_per_token_avg": round(mean(chars_per_token), 3) if chars_per_token else 0.0,
        "prompt_tokens": token_counts,}
def main() -> int:
    parser = argparse.ArgumentParser(
        description="benchmark counts"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("tests/results/benchmark_results.json"),
        help="Path to the benchmark_results.json file..",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON print out format",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate per-row prompt/token diagnostics.",
    )
    parser.add_argument(
        "--validate-usage",
        action="store_true",
        help="Cross-check tokenize counts",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=4096,
        help="context window checks",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Print retrieved chunks per question (ids, source, chars, preview).",
    )
    parser.add_argument(
        "--chunk-preview-chars",
        type=int,
        default=160,
        help="Max preview characters per chunk when --show-chunks is enabled.",
    )
    args = parser.parse_args()

    if not args.results.exists():
        raise FileNotFoundError(f"Benchmark results file not fuond: {args.results}")
    rows = load_results(args.results)

    usage_validation = None
    if args.verbose or args.validate_usage:
        _, _, usage_validation = count_prompt_tokens(
            rows,
            verbose=args.verbose,
            validate_usage=args.validate_usage,
            n_ctx=args.n_ctx,
        )

    summary = summarize(rows)

    chunk_details = None
    if args.show_chunks:
        chunk_details = _collect_chunk_details(rows, preview_chars=args.chunk_preview_chars)

    if usage_validation is not None:
        summary["usage_validation"] = usage_validation

    if chunk_details is not None:
        summary["retrieved_chunks"] = chunk_details

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Total benchmarks: {summary['total']}")
        print(f"Passed: {summary['passed']}")
        print(f"Pass rate: {summary['pass_rate']}")
        print(f"Average final score: {summary['avg_final']}")
        print(f"Average semantic score: {summary['avg_semantic']}")
        print(f"Average keyword score: {summary['avg_keyword']}")
        print(f"Average NLI score: {summary['avg_nli']}")
        print(f"Average prompt chars: {summary['prompt_chars_avg']}")
        print(f"Prompt chars min: {summary['prompt_chars_min']}")
        print(f"Prompt chars max: {summary['prompt_chars_max']}")
        print(f"Average prompt tokens: {summary['prompt_tokens_avg']}")
        print(f"Prompt tokens min: {summary['prompt_tokens_min']}")
        print(f"Prompt tokens max: {summary['prompt_tokens_max']}")
        print(f"Average chars per token: {summary['chars_per_token_avg']}")
        print(f"Prompt tokens list: {summary['prompt_tokens']}")
        if chunk_details is not None:
            print("\nRetrieved chunks per question:")
            for row in chunk_details:
                print(f"- {row['test_id']} | chunks={row['chunk_count']}")
                for chunk in row["chunks"]:
                    print(
                        "    "
                        f"[{chunk['position']}] id={chunk['chunk_id']} "
                        f"source={chunk['source']} chars={chunk['chars']}"
                    )
                    print(f"        {chunk['preview']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())