import sys
import tempfile

import torch
from fast_plaid import search

SEED = 42
N_DOCS = 200
DIM = 128
N_SAMPLES_KMEANS = 50  # force subsampling so randperm order matters


def make_data():
    rng = torch.Generator()
    rng.manual_seed(0)
    docs = [torch.randn(30 + (i % 20), DIM, generator=rng) for i in range(N_DOCS)]
    queries = torch.randn(20, 25, DIM, generator=rng)
    return docs, queries


def build(path, docs, seed=SEED):
    idx = search.FastPlaid(index=path, device="cpu")
    idx.create(documents_embeddings=docs, kmeans_niters=10, seed=seed, n_samples_kmeans=N_SAMPLES_KMEANS)
    return idx


def ids(results):
    return [[doc_id for doc_id, _ in q] for q in results]


def check(label, ok, msg=""):
    status = "PASS" if ok else f"FAIL — {msg}"
    print(f"{label}: {status}")
    return ok


def main():
    docs, queries = make_data()
    passed = []

    # same index, two searches
    with tempfile.TemporaryDirectory() as tmp:
        idx = build(tmp, docs)
        r1 = idx.search(queries_embeddings=queries, top_k=10)
        r2 = idx.search(queries_embeddings=queries, top_k=10)
        idx.close()
    passed.append(check("repeated search", ids(r1) == ids(r2)))

    # two independently-built indices, same seed
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        idx1 = build(a, docs)
        r1 = idx1.search(queries_embeddings=queries, top_k=10)
        idx1.close()
        idx2 = build(b, docs)
        r2 = idx2.search(queries_embeddings=queries, top_k=10)
        idx2.close()

    if ids(r1) != ids(r2):
        diff = next(
            (f"query {q}: {i1} vs {i2}" for q, (i1, i2) in enumerate(zip(ids(r1), ids(r2))) if i1 != i2),
            "unknown",
        )
    passed.append(check("same seed, two builds", ids(r1) == ids(r2), diff if ids(r1) != ids(r2) else ""))

    # sanity: different seed → different results
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        idx1 = build(a, docs, seed=SEED)
        r1 = idx1.search(queries_embeddings=queries, top_k=10)
        idx1.close()
        idx2 = build(b, docs, seed=SEED + 1)
        r2 = idx2.search(queries_embeddings=queries, top_k=10)
        idx2.close()
    passed.append(check("different seeds differ", ids(r1) != ids(r2), "seeds had no effect"))

    n_fail = passed.count(False)
    print(f"\n{len(passed) - n_fail}/{len(passed)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
