import json
import os
import tempfile

import torch
from fast_plaid import search

N_DOCS = 300
DOC_LEN = 64
DIM = 128
N_QUERIES = 10
QUERY_LEN = 20
TOP_K = 10
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_PATH = os.path.join(os.path.dirname(__file__), ".determinism_baseline.json")


def random_data():
    rng = torch.Generator()
    rng.manual_seed(0)
    docs = [torch.randn(DOC_LEN, DIM, generator=rng) for _ in range(N_DOCS)]
    queries = torch.randn(N_QUERIES, QUERY_LEN, DIM, generator=rng)
    return docs, queries


def build_and_search(path, docs, queries):
    idx = search.FastPlaid(index=path, device=DEVICE, deterministic=True)
    idx.create(documents_embeddings=docs, kmeans_niters=4, seed=SEED, n_samples_kmeans=100)
    results = idx.search(queries_embeddings=queries, top_k=TOP_K)
    idx.close()
    return [[d for d, _ in q] for q in results]


def main():
    docs, queries = random_data()
    print(f"device={DEVICE}  docs={N_DOCS}  doc_len={DOC_LEN}  queries={N_QUERIES}  top_k={TOP_K}")

    with tempfile.TemporaryDirectory() as tmp:
        ids = build_and_search(tmp, docs, queries)

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            baseline = json.load(f)
        mismatches = sum(1 for a, b in zip(ids, baseline) if a != b)
        if mismatches:
            print(f"FAIL: {mismatches}/{N_QUERIES} queries diverged from baseline")
            for q, (a, b) in enumerate(zip(ids, baseline)):
                if a != b:
                    print(f"  query {q}:\n    now:      {a}\n    baseline: {b}")
                    if q >= 2:
                        break
        else:
            print(f"PASS: all {N_QUERIES} queries match baseline ({RESULTS_PATH})")
    else:
        with open(RESULTS_PATH, "w") as f:
            json.dump(ids, f)
        print(f"baseline saved to {RESULTS_PATH} — rerun this script to check")


if __name__ == "__main__":
    main()
