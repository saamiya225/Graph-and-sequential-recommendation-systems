#!/usr/bin/env python3
import os
import argparse
import numpy as np
import scipy.sparse as sp

def main():
    parser = argparse.ArgumentParser(
        description="Build & cache the sparse adjacency from train.txt"
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

    # Path to train.txt
    data_dir = os.path.join('..', 'data', args.dataset)
    train_file = os.path.join(data_dir, 'train.txt')
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"train.txt not found in {data_dir}")

    users = []
    items = []
    # Read train.txt: lines of 'user item1 item2...'
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            for tok in parts[1:]:
                # support optional 'item:ts' tokens
                if ':' in tok:
                    i_str, _ = tok.split(':', 1)
                    i = int(i_str)
                else:
                    i = int(tok)
                users.append(u)
                items.append(i)

    users = np.array(users, dtype=np.int64)
    items = np.array(items, dtype=np.int64)

    # Determine number of users and items via max ID
    n_users = users.max() + 1
    n_items = items.max() + 1
    n_nodes = n_users + n_items

    # Build symmetric bipartite adjacency
    row_u = users
    col_u = items + n_users
    row_i = items + n_users
    col_i = users

    rows = np.concatenate([row_u, row_i])
    cols = np.concatenate([col_u, col_i])
    data = np.ones(len(rows), dtype=np.float32)

    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    # Save to cache
    os.makedirs(args.cache_dir, exist_ok=True)
    out_fp = os.path.join(args.cache_dir, f"{args.dataset}_sp_graph.npz")
    sp.save_npz(out_fp, adj)
    print(f"Wrote adjacency cache to {out_fp} ({len(users)} interactions)")

if __name__ == "__main__":
    main()
