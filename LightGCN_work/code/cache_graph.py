#!/usr/bin/env python3
"""
===============================================================================
⚠️ DEPRECATED SCRIPT – NOT USED IN FINAL PROJECT
===============================================================================

This script (`cache_graph.py`) was originally written when we experimented with 
a **PPR (Personalized PageRank) variant** of LightGCN. 
The idea was to precompute and cache the bipartite adjacency graph as a .npz 
file, to speed up training and enable custom propagation schemes.

👉 However, the PPR-based variant was **discarded from the project** because:
   - It did not improve results consistently across datasets,
   - It made the pipeline more complex,
   - Our final project focuses instead on Global, Global+MLP, and Fusion models.

Therefore:
   - You do NOT need this script to run the final models.
   - The adjacency graph for LightGCN is already built and cached automatically 
     inside the data loader (`getSparseGraph()` in `dataloader.py`).
   - This file is kept ONLY for archival/documentation purposes.

===============================================================================
"""

import os
import argparse
import numpy as np
import scipy.sparse as sp

def main():
    parser = argparse.ArgumentParser(
        description="(DEPRECATED) Build & cache sparse adjacency from train.txt "
                    "for the discarded PPR variant."
    )
    parser.add_argument(
        "--dataset", "-d", required=True,
        help="Name of the dataset folder under data/ (e.g. 'amazon-book')"
    )
    parser.add_argument(
        "--cache_dir", "-c", default="cache",
        help="Directory in which to save the .npz adjacency file"
    )
    args = parser.parse_args()

    # Path to train.txt (LightGCN format)
    data_dir = os.path.join('..', 'data', args.dataset)
    train_file = os.path.join(data_dir, 'train.txt')
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"train.txt not found in {data_dir}")

    users, items = [], []

    # -------------------------------------------------------------------------
    # train.txt format: each line is "user item1 item2 ..."
    # This loop parses interactions into user–item edge pairs.
    # Note: Supports optional "item:ts" (item with timestamp), but only the
    # item ID is kept.
    # -------------------------------------------------------------------------
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            for tok in parts[1:]:
                if ':' in tok:   # optional timestamp format
                    i_str, _ = tok.split(':', 1)
                    i = int(i_str)
                else:
                    i = int(tok)
                users.append(u)
                items.append(i)

    users = np.array(users, dtype=np.int64)
    items = np.array(items, dtype=np.int64)

    # -------------------------------------------------------------------------
    # Determine graph size: total users + total items.
    # Build a symmetric bipartite adjacency matrix.
    # -------------------------------------------------------------------------
    n_users = users.max() + 1
    n_items = items.max() + 1
    n_nodes = n_users + n_items

    row_u = users
    col_u = items + n_users
    row_i = items + n_users
    col_i = users

    rows = np.concatenate([row_u, row_i])
    cols = np.concatenate([col_u, col_i])
    data = np.ones(len(rows), dtype=np.float32)

    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    # -------------------------------------------------------------------------
    # Save adjacency to disk (for PPR experiments).
    # -------------------------------------------------------------------------
    os.makedirs(args.cache_dir, exist_ok=True)
    out_fp = os.path.join(args.cache_dir, f"{args.dataset}_sp_graph.npz")
    sp.save_npz(out_fp, adj)
    print(f"[DEPRECATED] Wrote adjacency cache to {out_fp} "
          f"({len(users)} interactions). "
          "This file is NOT used in the final project.")

if __name__ == "__main__":
    main()
