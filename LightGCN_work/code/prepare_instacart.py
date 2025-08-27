"""
prepare_instacart.py — Convert raw Instacart CSVs into LightGCN format.

This script:
- Reads `orders.csv` and `order_products__prior.csv`
- Keeps only 'prior' interactions
- Uses the last prior order per user as **test set**, earlier orders as **train**
- Supports subsampling users (subset_frac)
- Filters out users with fewer than `min_orders` prior orders
- Remaps user_id and product_id to consecutive indices
- Outputs `train.txt` and `test.txt` in LightGCN format:
    <user> <item1> <item2> ...

Outputs:
    out_dir/train.txt
    out_dir/test.txt

Example:
    python prepare_instacart.py \
        --raw_dir ../data/instacart/raw \
        --out_dir ../data/instacart \
        --subset_frac 0.1 \
        --min_orders 2
"""

import os
import argparse
import numpy as np
import pandas as pd


def prepare_instacart(
    raw_dir,
    out_dir,
    subset_frac=1.0,
    min_orders=2,
    seed=42,
):
    """
    Prepare Instacart data to LightGCN format (train.txt/test.txt).

    - Uses only 'prior' interactions from order_products__prior.csv.
    - For each user, the last prior order is used as test; others as train.
    - Optionally subsample users via subset_frac.
    - Robust handling of NaNs in raw CSVs.

    Args:
        raw_dir (str): folder with orders.csv + order_products__prior.csv
        out_dir (str): folder to write train.txt/test.txt
        subset_frac (float): fraction of users to keep (default=1.0 = all)
        min_orders (int): min # prior orders required per user
        seed (int): random seed for subsampling
    """
    os.makedirs(out_dir, exist_ok=True)

    orders_csv = os.path.join(raw_dir, "orders.csv")
    prior_csv  = os.path.join(raw_dir, "order_products__prior.csv")

    # ---- Load orders.csv ----
    print(f"[INFO] Reading {orders_csv}")
    orders = pd.read_csv(
        orders_csv,
        dtype={
            "order_id": "Int64",
            "user_id": "Int64",
            "eval_set": "string",
            "order_number": "Int64",
        },
        usecols=["order_id", "user_id", "eval_set", "order_number"],
        keep_default_na=True,
    )

    # Keep only 'prior' rows
    orders = orders[orders["eval_set"] == "prior"].copy()
    orders = orders.dropna(subset=["order_id", "user_id", "order_number"])
    orders["order_id"] = orders["order_id"].astype(np.int64)
    orders["user_id"] = orders["user_id"].astype(np.int64)
    orders["order_number"] = orders["order_number"].astype(np.int32)

    print(f"[INFO] Prior orders after cleaning: {len(orders)}")

    # ---- Load order_products__prior.csv ----
    print(f"[INFO] Reading {prior_csv}")
    prior = pd.read_csv(
        prior_csv,
        dtype={"order_id": "Int64", "product_id": "Int64"},
        usecols=["order_id", "product_id"],
        keep_default_na=True,
    )
    prior = prior.dropna(subset=["order_id", "product_id"])
    prior["order_id"] = prior["order_id"].astype(np.int64)
    prior["product_id"] = prior["product_id"].astype(np.int64)

    # ---- Merge with orders to attach user_id & order_number ----
    print("[INFO] Merging prior products with orders...")
    user_prods = prior.merge(
        orders[["order_id", "user_id", "order_number"]],
        on="order_id", how="inner"
    )
    del prior  # free memory

    # ---- Filter users with at least `min_orders` ----
    print(f"[INFO] Filtering users with >= {min_orders} prior orders...")
    orders_per_user = orders.groupby("user_id", as_index=False)["order_number"].max()
    orders_per_user.rename(columns={"order_number": "prior_count"}, inplace=True)
    valid_users = orders_per_user.loc[orders_per_user["prior_count"] >= min_orders, "user_id"].unique()
    user_prods = user_prods[user_prods["user_id"].isin(valid_users)].copy()
    print(f"[INFO] Users after filter: {user_prods['user_id'].nunique()}")

    # ---- Optional subsampling ----
    if subset_frac < 1.0:
        print(f"[INFO] Subsampling users: keeping {subset_frac*100:.1f}%")
        rng = np.random.default_rng(seed)
        all_users = user_prods["user_id"].unique()
        keep_count = max(1, int(len(all_users) * subset_frac))
        keep_users = rng.choice(all_users, size=keep_count, replace=False)
        user_prods = user_prods[user_prods["user_id"].isin(keep_users)].copy()
        print(f"[INFO] Kept users: {len(keep_users)}")

    # ---- Determine last order per user = test ----
    print("[INFO] Determining last prior order per user...")
    last_orders = user_prods.groupby("user_id", as_index=False)["order_number"].max()
    last_orders.rename(columns={"order_number": "last_order_number"}, inplace=True)

    user_prods = user_prods.merge(last_orders, on="user_id", how="left")
    user_prods["is_test"] = (user_prods["order_number"] == user_prods["last_order_number"])

    # ---- Remap IDs to consecutive ints ----
    print("[INFO] Remapping user_id & product_id to consecutive indices...")
    unique_users = user_prods["user_id"].unique()
    unique_items = user_prods["product_id"].unique()

    user2id = {u: idx for idx, u in enumerate(sorted(unique_users))}
    item2id = {p: idx for idx, p in enumerate(sorted(unique_items))}

    user_prods["uid"] = user_prods["user_id"].map(user2id).astype(np.int64)
    user_prods["iid"] = user_prods["product_id"].map(item2id).astype(np.int64)

    # ---- Build train/test maps ----
    print("[INFO] Building train/test interactions...")
    train_df = user_prods[~user_prods["is_test"]][["uid", "iid"]].drop_duplicates()
    test_df  = user_prods[user_prods["is_test"]][["uid", "iid"]].drop_duplicates()

    train_map = train_df.groupby("uid")["iid"].apply(list).to_dict()
    test_map  = test_df.groupby("uid")["iid"].apply(list).to_dict()

    users_with_train = set(train_map.keys())
    users_with_test  = set(test_map.keys())
    all_uids = sorted(users_with_train | users_with_test)

    # ---- Write train.txt / test.txt ----
    train_path = os.path.join(out_dir, "train.txt")
    test_path  = os.path.join(out_dir, "test.txt")
    print(f"[INFO] Writing LightGCN files to {out_dir}")

    with open(train_path, "w") as f_tr:
        for u in all_uids:
            items = train_map.get(u, [])
            if not items:
                continue
            items = sorted(set(items))
            f_tr.write(str(u) + " " + " ".join(str(i) for i in items) + "\n")

    with open(test_path, "w") as f_te:
        for u in all_uids:
            items = test_map.get(u, [])
            if not items:
                continue
            items = sorted(set(items))
            f_te.write(str(u) + " " + " ".join(str(i) for i in items) + "\n")

    # ---- Report summary ----
    print("[OK] Done.")
    print(f"  Users total        : {len(all_uids)}")
    print(f"  Users with train   : {len(train_map)}")
    print(f"  Users with test    : {len(test_map)}")
    print(f"  Train interactions : {int(train_df.shape[0])}")
    print(f"  Test  interactions : {int(test_df.shape[0])}")
    print(f"  Num items          : {len(item2id)}")
    print(f"  Saved: {train_path}")
    print(f"         {test_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare Instacart dataset for LightGCN")
    ap.add_argument("--raw_dir", type=str, default="../data/instacart/raw",
                    help="Folder with raw Instacart CSVs (orders.csv + order_products__prior.csv)")
    ap.add_argument("--out_dir", type=str, default="../data/instacart",
                    help="Output folder for train.txt/test.txt")
    ap.add_argument("--subset_frac", type=float, default=1.0,
                    help="Fraction of users to keep (e.g., 0.1 = 10%)")
    ap.add_argument("--min_orders", type=int, default=2,
                    help="Minimum prior orders required per user")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for subsampling")
    args = ap.parse_args()

    prepare_instacart(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        subset_frac=args.subset_frac,
        min_orders=args.min_orders,
        seed=args.seed,
    )
