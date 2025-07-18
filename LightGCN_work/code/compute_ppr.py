#!/usr/bin/env python3
import os
import argparse
import numpy as np
import scipy.sparse as sp

def compute_ppr_weights(adj_coo, alpha=0.15, K=3):
    """
    Compute personalized PageRank–based layer weights.

    adj_coo: scipy.sparse.coo_matrix, shape [N, N]
        Raw adjacency for undirected graph (symmetric bipartite).
    alpha: float
        Restart probability (default 0.15).
    K: int
        Number of propagation steps (e.g. your LightGCN n_layers).

    Returns
    -------
    W: np.ndarray, shape [N, K+1]
        Row-normalized PPR weights for each node across distances 0..K.
    """
    # Build row-stochastic transition matrix T
    row_sum = np.array(adj_coo.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    inv_row = sp.diags(1.0 / row_sum)
    T = inv_row.dot(adj_coo)  # each row sums to 1

    N = T.shape[0]
    # Initialize Pk = T^0 (identity)
    Pk = sp.eye(N, format='csr')
    W = np.zeros((N, K+1), dtype=np.float32)

    # Accumulate α * (1-α)^k T^k contributions
    for k in range(K+1):
        # α * row sums of Pk
        W[:, k] = alpha * Pk.sum(axis=1).A.ravel()
        # Next: multiply by (1-α) T
        Pk = (1.0 - alpha) * T.dot(Pk)

    # Normalize rows to sum to 1
    row_totals = W.sum(axis=1, keepdims=True) + 1e-12
    W = W / row_totals
    return W

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and cache PPR-based layer weights for LightGCN"
    )
    parser.add_argument(
        "--dataset", "-d", required=True,
        help="Dataset name matching cache graph file, e.g. 'amazon-book'"
    )
    parser.add_argument(
        "--cache_dir", "-c", default="cache",
        help="Directory containing <dataset>_sp_graph.npz and to write PPR weights"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.15,
        help="Restart probability for PPR (default: 0.15)"
    )
    parser.add_argument(
        "--K", type=int, default=3,
        help="Number of propagation steps (default: 3)"
    )
    args = parser.parse_args()

    # Load cached adjacency
    sp_fp = os.path.join(args.cache_dir, f"{args.dataset}_sp_graph.npz")
    if not os.path.exists(sp_fp):
        raise FileNotFoundError(f"Adjacency cache not found: {sp_fp}")
    print(f"Loading adjacency from {sp_fp}...")
    A = sp.load_npz(sp_fp).tocoo()

    # Compute and save PPR weights
    print(f"Computing PPR weights: alpha={args.alpha}, K={args.K}...")
    W = compute_ppr_weights(A, alpha=args.alpha, K=args.K)
    out_fp = os.path.join(args.cache_dir, f"{args.dataset}_ppr_weights.npy")
    np.save(out_fp, W)
    print(f"Wrote PPR weights to {out_fp}")
