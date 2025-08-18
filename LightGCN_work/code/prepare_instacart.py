# prepare_instacart.py  (drop into LightGCN_work/code/)
# Generates train.txt and test.txt for the "instacart" dataset folder.

import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "data", "instacart", "raw")
OUT = os.path.join(ROOT, "data", "instacart")

os.makedirs(OUT, exist_ok=True)

orders_fp  = os.path.join(RAW, "orders.csv")
prior_fp   = os.path.join(RAW, "order_products__prior.csv")
train_fp   = os.path.join(RAW, "order_products__train.csv")
products_fp = os.path.join(RAW, "products.csv")  # optional

print("Loading raw Instacart data...")
usecols_orders = ["order_id", "user_id", "order_number"]
orders = pd.read_csv(orders_fp, usecols=usecols_orders, dtype=int)

# We only need order_id and product_id from the order_products files
usecols_op = ["order_id", "product_id"]
op_prior = pd.read_csv(prior_fp, usecols=usecols_op, dtype=int)
op_train = pd.read_csv(train_fp, usecols=usecols_op, dtype=int)

# Combine prior+train interactions (both are historical)
op_all = pd.concat([op_prior, op_train], ignore_index=True)

print("Encoding user_id and product_id...")

# --- Reindex users/products to contiguous ids (0..N-1) ---
# User ids from orders table
u_old = orders["user_id"].unique()
u_old.sort()
u_map = pd.Series(index=u_old, data=np.arange(len(u_old)))

# Product ids from all order_products
p_old = op_all["product_id"].unique()
p_old.sort()
p_map = pd.Series(index=p_old, data=np.arange(len(p_old)))

# Apply mappings
orders["uid"] = orders["user_id"].map(u_map).astype(int)
op_all["pid"] = op_all["product_id"].map(p_map).astype(int)

# Merge to get uid per order row
user_products = op_all.merge(orders[["order_id", "uid", "order_number"]], on="order_id", how="left")

print("Splitting train/test...")

# Find each user's last order by order_number
idx = orders.groupby("uid")["order_number"].idxmax()
last_orders = orders.loc[idx, "order_id"]       # Series of last order_ids per uid
last_order_ids = set(last_orders.to_numpy())    # <-- ensure list-like

# Mark interactions from last orders as test
user_products["is_test"] = user_products["order_id"].isin(last_order_ids)

# Build per-user item lists
train_lines = []
test_lines  = []

# Group by user and collect item ids for train/test
g = user_products.groupby("uid")

for uid, df in g:
    train_items = df.loc[~df["is_test"], "pid"].unique()
    test_items  = df.loc[df["is_test"],  "pid"].unique()
    if len(train_items) > 0:
        train_lines.append(str(uid) + " " + " ".join(map(str, train_items)))
    if len(test_items) > 0:
        test_lines.append(str(uid) + " " + " ".join(map(str, test_items)))

# Write files in LightGCN format
train_out = os.path.join(OUT, "train.txt")
test_out = os.path.join(OUT, "test.txt")
with open(train_out, "w") as f:
    f.write("\n".join(train_lines))
with open(test_out, "w") as f:
    f.write("\n".join(test_lines))

print(f"Wrote {len(train_lines)} users to {train_out}")
print(f"Wrote {len(test_lines)} users to {test_out}")
print("Done.")
