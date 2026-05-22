"""Benchmark deterministic vs non-deterministic mode.

Measures the overhead of `FastPlaid(deterministic=True)`, which routes
argmax/topk/quantile through sort-based equivalents and disables the Triton
k-means kernel. Uses synthetic embeddings so the benchmark has no model
dependency.

Usage:
    python docs/benchmark/deterministic_benchmark.py
    python docs/benchmark/deterministic_benchmark.py --n-docs 50000 --doc-len 200 --n-queries 200
"""

import argparse
import shutil
import statistics
import tempfile
import time

import torch
from fast_plaid import search


def synth_data(n_docs: int, doc_len: int, n_queries: int, query_len: int, dim: int, seed: int):
    rng = torch.Generator().manual_seed(seed)
    docs = [torch.randn(doc_len, dim, generator=rng) for _ in range(n_docs)]
    queries = torch.randn(n_queries, query_len, dim, generator=rng)
    return docs, queries


def time_run(deterministic: bool, docs, queries, device: str, top_k: int, seed: int):
    """Build an index and run a search; return (index_time, search_time)."""
    tmp = tempfile.mkdtemp()
    try:
        idx = search.FastPlaid(index=tmp, device=device, deterministic=deterministic)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        idx.create(documents_embeddings=docs, kmeans_niters=4, seed=seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        index_time = time.perf_counter() - t0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        idx.search(queries_embeddings=queries, top_k=top_k)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        search_time = time.perf_counter() - t0

        idx.close()
        return index_time, search_time
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def trimmed(samples: list[float]) -> list[float]:
    """Drop the single highest and lowest sample to discount JIT/launch outliers."""
    if len(samples) <= 2:
        return samples
    return sorted(samples)[1:-1]


def fmt(name: str, samples: list[float]) -> str:
    t = trimmed(samples)
    return (
        f"{name:>14}: median={statistics.median(t)*1000:8.1f} ms  "
        f"mean={statistics.mean(t)*1000:8.1f} ms  "
        f"min={min(samples)*1000:8.1f} ms  max={max(samples)*1000:8.1f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-docs", type=int, default=20_000)
    parser.add_argument("--doc-len", type=int, default=180)
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--query-len", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Device:     {args.device}")
    print(f"Docs:       {args.n_docs} × {args.doc_len} × {args.dim}")
    print(f"Queries:    {args.n_queries} × {args.query_len} × {args.dim}, top_k={args.top_k}")
    print(f"Repeats:    {args.repeats} (min/max trimmed for median + mean)")
    if args.device == "cpu":
        print("\nNote: CPU paths are already deterministic; the flag is a no-op there.\n")

    docs, queries = synth_data(args.n_docs, args.doc_len, args.n_queries, args.query_len, args.dim, args.seed)

    # Two warm-up passes per mode at this exact workload to amortize CUDA init
    # and per-shape kernel JIT before any timing starts.
    print("warming up...")
    for deterministic in (False, True):
        for _ in range(2):
            time_run(deterministic, docs, queries, args.device, args.top_k, args.seed)

    runs = {True: {"index": [], "search": []}, False: {"index": [], "search": []}}
    for r in range(args.repeats):
        for deterministic in (False, True):
            it, st = time_run(deterministic, docs, queries, args.device, args.top_k, args.seed)
            runs[deterministic]["index"].append(it)
            runs[deterministic]["search"].append(st)
            tag = "deterministic" if deterministic else "default"
            print(f"  [{r+1}/{args.repeats}] {tag:14s} index={it*1000:8.1f} ms  search={st*1000:8.1f} ms")

    def overhead(metric: str) -> float:
        det = statistics.median(trimmed(runs[True][metric]))
        base = statistics.median(trimmed(runs[False][metric]))
        return (det / base - 1) * 100

    print("\n--- index creation ---")
    print(fmt("default", runs[False]["index"]))
    print(fmt("deterministic", runs[True]["index"]))
    print(f"  → deterministic overhead (trimmed median): {overhead('index'):+.1f}%")

    print("\n--- search ---")
    print(fmt("default", runs[False]["search"]))
    print(fmt("deterministic", runs[True]["search"]))
    print(f"  → deterministic overhead (trimmed median): {overhead('search'):+.1f}%")


if __name__ == "__main__":
    main()
