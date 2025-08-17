import pandas as pd
import os

RAW_DIR = "../data/instacart/raw"
OUT_DIR = "../data/instacart"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading raw Instacart data...")
orders = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))
prior = pd.read_csv(os.path.join(RAW_DIR, "order_products__prior.csv"))
train = pd.read_csv(os.path.join(RAW_DIR, "order_products__train.csv"))

# Merge prior and train
order_products = pd.concat([prior, train], axis=0, ignore_index=True)

# Merge with orders to get user_id
orders = orders[["order_id", "user_id"]]
user_products = order_products.merge(orders, on="order_id")[["user_id", "product_id", "order_id"]]

# Encode IDs
print("Encoding user_id and product_id...")
user2id = {u: i for i, u in enumerate(user_products["user_id"].unique())}
item2id = {p: i for i, p in enumerate(user_products["product_id"].unique())}

user_products["user_id"] = user_products["user_id"].map(user2id)
user_products["product_id"] = user_products["product_id"].map(item2id)

# Train/test split: last order per user goes to test
print("Splitting train/test...")
last_orders = orders.groupby("user_id")["order_id"].max().to_dict()
user_products["is_test"] = user_products["order_id"].isin(last_orders.values)

train_df = user_products[user_products["is_test"] == False][["user_id", "product_id"]]
test_df  = user_products[user_products["is_test"] == True][["user_id", "product_id"]]

# Save outputs
print("Saving preprocessed files...")
train_df.to_csv(os.path.join(OUT_DIR, "train.txt"), index=False, sep="\t", header=False)
test_df.to_csv(os.path.join(OUT_DIR, "test.txt"), index=False, sep="\t", header=False)

print(f"✅ Done. Train interactions={len(train_df)}, Test interactions={len(test_df)}")
