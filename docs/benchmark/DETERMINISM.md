# Deterministic mode benchmark

This page summarizes the cost of `FastPlaid(deterministic=True)` — the flag
that routes index creation and search through CUDA-deterministic kernels so
two builds with the same seed produce bit-identical results.

## What changes when `deterministic=True`

| Component | Default (CUDA) | Deterministic |
|---|---|---|
| K-means | Triton kernel (non-deterministic atomics) | Standard PyTorch K-means |
| `argmax` / `topk` (codec, IVF probe, candidate prune) | `argmax` / `topk` (memory-order tie-breaking) | Sort + index/narrow |
| `kthvalue` quantile (codec setup, threshold update) | `kthvalue` (no deterministic CUDA impl) | Full-sort quantile |
| Strided-tensor strides | Random subsample → quantile | Full sort + quantile (already deterministic in this PR) |

CPU is already deterministic; the flag is a no-op for correctness on CPU but
still routes through the slower sort-based ops, so leave it off there.

## Reproducing

```bash
python docs/benchmark/deterministic_benchmark.py \
    --device cuda:0 --n-docs 50000 --doc-len 200 \
    --n-queries 200 --top-k 100 --repeats 5
```

Each workload runs:

1. 2 warm-up builds per mode (amortizes per-shape kernel JIT).
2. `--repeats` timed builds per mode, alternating default / deterministic.
3. Reports trimmed-median (drop min + max) plus untrimmed min / max so
   variance is visible.

## Results — NVIDIA H200, torch 2.7.1+cu126

Workload key: `N docs × tokens-per-doc, k = top_k`. Default queries unless
noted: 100 queries × 32 tokens.

| Workload | Shape | Index Δ | Search Δ |
|---|---|---:|---:|
| small | 5K × 120, k=10 | −12.9% | +26.4% |
| medium | 20K × 180, k=100 | +8.8% | −3.6% |
| large | 50K × 200, k=100 | +12.9% | +7.2% |
| long-doc | 10K × 500, k=100 | **+73.2%** | −19.1% |
| high-topk | 20K × 180, k=1000 | −5.7% | +20.5% |
| xlarge | 100K × 200, k=100 | **+65.4%** | +14.0% |
| many-queries | 50K × 200, 1000 queries, k=100 | +12.2% | −2.9% |
| long-doc-xl | 30K × 500, k=100 | +36.1% | −6.5% |
| xxlarge | 250K × 200, k=100 | +51.8% | **−20.3%** |

Δ is `(deterministic − default) / default` on the trimmed-median wall clock.

## Reading the numbers

**Index creation overhead is bimodal, not smoothly N-scaling.**

- ~10–15% on workloads where the standard K-means kernel is roughly
  competitive with Triton (small, medium, large, many-queries).
- ~35–70% when Triton K-means would have been a substantial win (long
  documents, 100K+ documents). Disabling Triton — required because its
  atomics are non-deterministic — is the dominant cost, not the
  sort-based codec ops.

The worst single case in this sweep is `long-doc` (10K × 500 tokens) at
+73%. Per-token cost matters more than `N` alone.

**Search overhead is workload-dependent, not a fixed tax.**

- Small / medium: roughly ±10% depending on noise.
- High top-k (k=1000): +20.5%, since full sort doesn't benefit from the
  `k` cap that `topk` does.
- Large scale (xxlarge, 250K docs): −20.3%. With very large candidate
  sets, sort is competitive with topk on H200 and the deterministic path
  edges ahead — possibly because of more regular memory traffic.

If you're picking sort-vs-topk by hand for production, benchmark on your
actual `n_full_scores`; that's the knob that flips the result.

## Bit-identity check

The flag is for *reproducibility*, not speed. The accompanying test
(`tests/test.py::TestDeterministic::test_two_builds_same_seed_match`)
confirms two `deterministic=True` builds with the same seed return
identical doc IDs and identical scores on CUDA.

## Raw log

Full sweep output (sample-by-sample) is in
[`../../scripts/deterministic_benchmark.log`](../../scripts/deterministic_benchmark.log)
when present locally — it's a personal artifact, not committed.
