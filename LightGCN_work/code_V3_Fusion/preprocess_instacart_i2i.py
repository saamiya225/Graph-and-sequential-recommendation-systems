"""
preprocess_instacart_i2i.py — Build item–item adjacency graph for Instacart (or similar).

This script:
- Reads user baskets from train.txt (LightGCN format: `user item1 item2 ...`)
- Builds an item–item co-occurrence graph with optional weighting:
    * cooc    : raw co-occurrence counts
    * jaccard : Jaccard similarity
    * pmi     : positive PMI
- Keeps top-K neighbors per item (sparse)
- Symmetrizes & degree-normalizes adjacency
- Saves the resulting CSR matrix to .npz for use in model.py (LightGCN item–item fusion)

Outputs:
    - <data_root>/<out> (default: i2i_adj.npz)

Example:
    python preprocess_instacart_i2i.py --data_root ../data/instacart --topk 50 --weight jaccard
"""

import os
import argparse
import math
import heapq
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm
from itertools import combinations


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def infer_n_items_from_files(train_path: str, test_path: str = None) -> int:
    """
    Infer item vocabulary size = max item id + 1.
    Scans train.txt (and optionally test.txt).

    Format per line: user item1 item2 ...
    """
    max_item = -1
    for path in [train_path, test_path]:
        if path is None or not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                items = [int(x) for x in parts[1:]]
                if items:
                    max_item = max(max_item, max(items))
    return max_item + 1


# ----------------------------------------------------------------------
# Main builder
# ----------------------------------------------------------------------
def build_item_item(
    train_path: str,
    n_items: int = None,
    topk: int = 50,
    weight: str = "cooc",
    min_basket: int = 1
) -> csr_matrix:
    """
    Build a symmetric item–item adjacency graph from train.txt.

    Args:
        train_path : path to train.txt
        n_items    : total number of items; inferred if None
        topk       : keep top-k neighbors per item
        weight     : 'cooc' | 'jaccard' | 'pmi'
        min_basket : skip baskets with fewer than this many items

    Returns:
        scipy.sparse.csr_matrix (degree-normalized, float32)
    """
    # ---- infer number of items ----
    if n_items is None:
        n_items = infer_n_items_from_files(train_path)

    # Co-occurrence counts and item basket frequencies
    cooc = defaultdict(lambda: defaultdict(float))
    item_deg = np.zeros(n_items, dtype=np.int64)  # baskets per item
    total_baskets = 0

    # Pass 1: accumulate counts
    with open(train_path, 'r') as f:
        for line in tqdm(f, desc="Reading train"):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            items = list(sorted(set(int(x) for x in parts[1:])))
            if len(items) < min_basket:
                continue
            total_baskets += 1

            # item frequency
            for it in items:
                item_deg[it] += 1

            # update co-occurrence for all unordered pairs
            for i, j in combinations(items, 2):
                cooc[i][j] += 1.0
                cooc[j][i] += 1.0

    # ---- transform weights ----
    if weight.lower() == "jaccard":
        for i, neigh in cooc.items():
            di = item_deg[i]
            for j in list(neigh.keys()):
                cij = neigh[j]
                dj = item_deg[j]
                denom = float(di + dj - cij)
                neigh[j] = 0.0 if denom <= 0.0 else (cij / denom)

    elif weight.lower() == "pmi":
        total = float(total_baskets) if total_baskets > 0 else 1.0
        for i, neigh in cooc.items():
            di = float(item_deg[i])
            for j in list(neigh.keys()):
                cij = float(neigh[j])
                dj = float(item_deg[j])
                denom = di * dj
                if denom <= 0.0:
                    neigh[j] = 0.0
                else:
                    pmi = math.log((cij * total) / denom + 1e-12)
                    neigh[j] = max(pmi, 0.0)

    # ---- build CSR rows ----
    indptr, indices, data = [0], [], []
    for i in tqdm(range(n_items), desc="Build rows"):
        neigh = cooc.get(i, {})
        if not neigh:
            indptr.append(indptr[-1])
            continue
        if len(neigh) > topk:
            top = heapq.nlargest(topk, neigh.items(), key=lambda x: x[1])
        else:
            top = list(neigh.items())

        cols  = [j for j, _ in top]
        vals  = [float(v) for _, v in top]
        indices.extend(cols)
        data.extend(vals)
        indptr.append(indptr[-1] + len(cols))

    sp = csr_matrix(
        (np.asarray(data, dtype=np.float32),
         np.asarray(indices, dtype=np.int64),
         np.asarray(indptr, dtype=np.int64)),
        shape=(n_items, n_items),
        dtype=np.float32
    )

    # ---- symmetrize ----
    sp = sp.maximum(sp.transpose())

    # ---- degree normalization: D^{-1/2} A D^{-1/2} ----
    deg = np.ravel(sp.sum(axis=1)).astype(np.float32)
    deg[deg == 0.0] = 1.0
    inv_sqrt = 1.0 / np.sqrt(deg)
    sp = sp.multiply(inv_sqrt[:, None])
    sp = sp.multiply(inv_sqrt[None, :])

    return sp.tocsr()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="../data/instacart",
                    help="Folder containing train.txt/test.txt")
    ap.add_argument("--train_file", type=str, default="train.txt")
    ap.add_argument("--test_file", type=str, default="test.txt")
    ap.add_argument("--out", type=str, default="i2i_adj.npz",
                    help="Output filename under data_root")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--weight", type=str, default="cooc",
                    choices=["cooc", "jaccard", "pmi"],
                    help="Edge weighting scheme")
    ap.add_argument("--min_basket", type=int, default=1,
                    help="Skip baskets with < min_basket items")
    ap.add_argument("--n_items", type=int, default=None,
                    help="Force #items (overrides inference)")
    args = ap.parse_args()

    train_path = os.path.join(args.data_root, args.train_file)
    test_path  = os.path.join(args.data_root, args.test_file)
    os.makedirs(args.data_root, exist_ok=True)

    n_items = args.n_items or infer_n_items_from_files(train_path, test_path)

    sp = build_item_item(
        train_path=train_path,
        n_items=n_items,
        topk=args.topk,
        weight=args.weight,
        min_basket=args.min_basket
    )
    out_path = os.path.join(args.data_root, args.out)
    save_npz(out_path, sp)
    print(f"[OK] saved i2i graph to {out_path}; nnz={sp.nnz}, shape={sp.shape}")
