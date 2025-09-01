#!/usr/bin/env python3
"""
⚠️ DEPRECATED SCRIPT — NOT USED IN FINAL PROJECT ⚠️

This script was originally written to compute Personalized PageRank (PPR) 
weights for each node, to experiment with a PPR-weighted LightGCN variant.

- Input: Cached sparse adjacency graph (<dataset>_sp_graph.npz).
- Output: Row-normalized PPR weight matrix (<dataset>_ppr_weights.npy).
- Idea: At each propagation layer k (0..K), apply PPR-based weights instead of uniform averaging.

We have since decided to **discard the PPR variant** from our project.
The final paper & experiments do **NOT** use PPR weights.

This file is kept in the repository **only for historical reference**.
It is safe to ignore and will not affect the main training pipeline.
"""

import os
import argparse
import numpy as np
import scipy.sparse as sp


def compute_ppr_weights(adj_coo, alpha=0.15, K=3):
    """
    Compute Personalized PageRank (PPR)–based propagation weights.

    Parameters
    ----------
    adj_coo : scipy.sparse.coo_matrix, shape [N, N]
        Symmetric adjacency matrix (users+items).
    alpha : float
        Restart probability (default = 0.15).
    K : int
        Number of propagation steps / LightGCN layers.

    Returns
    -------
    W : np.ndarray, shape [N, K+1]
        Row-normalized PPR weights for each node at distances 0..K.
    """
    # Build row-stochastic transition matrix T (each row sums to 1)
    row_sum = np.array(adj_coo.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    inv_row = sp.diags(1.0 / row_sum)
    T = inv_row.dot(adj_coo)

    N = T.shape[0]
    Pk = sp.eye(N, format='csr')   # T^0 = identity
    W = np.zeros((N, K+1), dtype=np.float32)

    # Accumulate contributions: α * (1-α)^k T^k
    for k in range(K+1):
        W[:, k] = alpha * Pk.sum(axis=1).A.ravel()
        Pk = (1.0 - alpha) * T.dot(Pk)

    # Normalize rows to sum to 1
    row_totals = W.sum(axis=1, keepdims=True) + 1e-12
    W = W / row_totals
    return W


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="(Deprecated) Compute and cache PPR-based layer weights for LightGCN"
    )
    parser.add_argument("--dataset", "-d", required=True,
                        help="Dataset name matching cache graph file (e.g. 'amazon-book')")
    parser.add_argument("--cache_dir", "-c", default="cache",
                        help="Directory containing <dataset>_sp_graph.npz and saving output")
    parser.add_argument("--alpha", type=float, default=0.15,
                        help="Restart probability (default=0.15)")
    parser.add_argument("--K", type=int, default=3,
                        help="Number of propagation steps (default=3)")
    args = parser.parse_args()

    # Load adjacency
    sp_fp = os.path.join(args.cache_dir, f"{args.dataset}_sp_graph.npz")
    if not os.path.exists(sp_fp):
        raise FileNotFoundError(f"Adjacency cache not found: {sp_fp}")
    print(f"[DEPRECATED] Loading adjacency from {sp_fp}...")
    A = sp.load_npz(sp_fp).tocoo()

    # Compute and save PPR weights
    print(f"[DEPRECATED] Computing PPR weights (alpha={args.alpha}, K={args.K})...")
    W = compute_ppr_weights(A, alpha=args.alpha, K=args.K)
    out_fp = os.path.join(args.cache_dir, f"{args.dataset}_ppr_weights.npy")
    np.save(out_fp, W)
    print(f"[DEPRECATED] Wrote PPR weights to {out_fp}")
