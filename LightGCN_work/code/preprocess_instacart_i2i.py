import os
import argparse
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm
from collections import defaultdict

def build_item_item(train_path, n_items=None, topk=50):
    """
    Build item-item CSR adjacency from train.txt (user followed by item list).
    Uses co-occurrence across the same user's history/basket.
    """
    # First pass: determine n_items if not provided
    if n_items is None:
        max_item = -1
        with open(train_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                items = [int(x) for x in parts[1:]]
                if items:
                    max_item = max(max_item, max(items))
        n_items = max_item + 1

    # Accumulate co-occurrence
    cooc = defaultdict(lambda: defaultdict(float))
    with open(train_path, 'r') as f:
        for line in tqdm(f, desc="Reading train"):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            items = list(sorted(set(int(x) for x in parts[1:])))
            L = len(items)
            for i in range(L):
                ii = items[i]
                for j in range(L):
                    jj = items[j]
                    if ii == jj:
                        continue
                    cooc[ii][jj] += 1.0

    # Build sparse rows with top-k neighbors per item
    indptr = [0]
    indices = []
    data = []
    for i in tqdm(range(n_items), desc="Build rows"):
        neigh = cooc.get(i, {})
        if len(neigh) == 0:
            indptr.append(indptr[-1])
            continue
        # keep top-k
        pairs = sorted(neigh.items(), key=lambda x: -x[1])[:topk]
        cols  = [j for j,_ in pairs]
        vals  = [v for _,v in pairs]
        indices.extend(cols)
        data.extend(vals)
        indptr.append(indptr[-1] + len(cols))

    sp = csr_matrix((np.array(data, dtype=np.float32),
                     np.array(indices, dtype=np.int32),
                     np.array(indptr, dtype=np.int32)),
                    shape=(n_items, n_items))

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    deg = np.ravel(sp.sum(axis=1))
    deg[deg == 0.0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    sp = sp.multiply(d_inv_sqrt[:, None])
    sp = sp.multiply(d_inv_sqrt[None, :])

    return sp

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="../data/instacart",
                    help="Folder containing train.txt/test.txt in LightGCN format")
    ap.add_argument("--train_file", type=str, default="train.txt")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--out", type=str, default="i2i_adj.npz")
    args = ap.parse_args()

    train_path = os.path.join(args.data_root, args.train_file)
    sp = build_item_item(train_path, topk=args.topk)
    out_path = os.path.join(args.data_root, args.out)
    os.makedirs(args.data_root, exist_ok=True)
    save_npz(out_path, sp)
    print(f"[OK] saved i2i to {out_path} with nnz={sp.nnz}")
