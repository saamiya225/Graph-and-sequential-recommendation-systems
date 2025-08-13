#!/usr/bin/env python3
"""
Convert Instacart (2017 Kaggle) into LightGCN-style train/val/test.

Outputs:
  - train.txt / val.txt / test.txt   (each line: "uid iid")
  - user_map.csv / item_map.csv
"""
import os, argparse, pandas as pd, np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, required=True,
                    help="Folder with orders.csv, order_products__prior.csv, order_products__train.csv")
    ap.add_argument("--out_dir", type=str, default="./data/instacart")
    ap.add_argument("--min_user_orders", type=int, default=3)
    ap.add_argument("--min_item_freq", type=int, default=5)
    ap.add_argument("--sample_users", type=int, default=0, help="Optional subsample after filtering")
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument("--no_val", action="store_true", help="Write only train/test (no val)")
    ap.add_argument("--dedup_basket", action="store_true", help="Remove duplicate items within an order")
    return ap.parse_args()

def write_pairs(pairs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for u, i in pairs:
            f.write(f"{u} {i}\n")

def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    orders = pd.read_csv(os.path.join(a.raw_dir, "orders.csv"),
                         usecols=["order_id","user_id","order_number"])
    p1 = pd.read_csv(os.path.join(a.raw_dir, "order_products__prior.csv"),
                     usecols=["order_id","product_id"])
    p2 = pd.read_csv(os.path.join(a.raw_dir, "order_products__train.csv"),
                     usecols=["order_id","product_id"])
    op = pd.concat([p1, p2], ignore_index=True)

    df = op.merge(orders, on="order_id", how="inner")[["user_id","order_number","product_id"]]

    # user min orders
    order_counts = df.drop_duplicates(["user_id","order_number"]).groupby("user_id").size()
    keep_users = order_counts[order_counts >= a.min_user_orders].index
    df = df[df["user_id"].isin(keep_users)]

    # de-dup within basket
    if a.dedup_basket:
        df = df.drop_duplicates(["user_id","order_number","product_id"])

    # item min freq
    item_freq = df.groupby("product_id").size()
    keep_items = item_freq[item_freq >= a.min_item_freq].index
    df = df[df["product_id"].isin(keep_items)]

    # sort and group
    df = df.sort_values(["user_id","order_number"])
    baskets = df.groupby(["user_id","order_number"])["product_id"].apply(list).reset_index()

    # optional subsample users
    users = baskets["user_id"].drop_duplicates().tolist()
    if a.sample_users > 0 and a.sample_users < len(users):
        users = set(rng.choice(users, size=a.sample_users, replace=False))
        baskets = baskets[baskets["user_id"].isin(users)]

    # remap ids
    users_final = baskets["user_id"].drop_duplicates().tolist()
    item_set = set()
    for prods in baskets["product_id"]:
        item_set.update(prods)
    items_final = sorted(item_set)

    user2id = {u:i for i,u in enumerate(users_final)}
    item2id = {it:i for i,it in enumerate(items_final)}

    # split per user
    train, val, test = [], [], []
    for u, g in baskets.groupby("user_id"):
        g = g.sort_values("order_number")
        uid = user2id[u]
        seq = []
        for prods in g["product_id"].tolist():
            if a.dedup_basket: prods = list(set(prods))
            seq.append([item2id[p] for p in prods if p in item2id])
        if len(seq) >= 3 and not a.no_val:
            for b in seq[:-2]:
                for i in b: train.append((uid, i))
            for i in seq[-2]: val.append((uid, i))
            for i in seq[-1]: test.append((uid, i))
        elif len(seq) >= 2:
            for b in seq[:-1]:
                for i in b: train.append((uid, i))
            for i in seq[-1]: test.append((uid, i))
        elif len(seq) == 1:
            for i in seq[0]: test.append((uid, i))

    write_pairs(train, os.path.join(a.out_dir, "train.txt"))
    if not a.no_val:
        write_pairs(val,   os.path.join(a.out_dir, "val.txt"))
    write_pairs(test,  os.path.join(a.out_dir, "test.txt"))

    pd.DataFrame({"orig_user_id": users_final,
                  "uid": [user2id[u] for u in users_final]}).to_csv(
        os.path.join(a.out_dir, "user_map.csv"), index=False)
    pd.DataFrame({"orig_product_id": items_final,
                  "iid": [item2id[i] for i in items_final]}).to_csv(
        os.path.join(a.out_dir, "item_map.csv"), index=False)

    print(f"[OK] Wrote to {a.out_dir}")
    print(f"users={len(users_final)} items={len(items_final)} "
          f"train={len(train)} {'val='+str(len(val)) if not a.no_val else ''} test={len(test)}")

if __name__ == "__main__":
    main()
