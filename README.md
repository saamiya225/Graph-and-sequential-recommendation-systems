# Limitations of User-Item Interactions in Graph Recommenders.

Lightweight experiments in graph-based recommendation centered on LightGCN, with practical extensions:

Variants

V0 - Baseline
Standard LightGCN cloned from https://github.com/gusye1234/LightGCN-PyTorch.git.

V1 — Global Smoothing
Extends V0 with global smoothing.

V2 — MLP Scoring
Extends V0 with an MLP-based scoring head instead of a plain dot product.
This lets the model learn a richer interaction function.

V3 — Fusion
Extends V2 with fusion mechanisms-

Popularity-aware gating (mixes item embedding + popularity embedding)

Item–item adjacency smoothing

Preprocessing scripts for the Instacart dataset.

The codebase includes training, evaluation, preprocessing (Instacart), checkpointing, and logging (CSV + TensorBoard).

Model evolution:

We started with the base code of LightGCN, and implemented different versions to test our research questions. We went from V0 -> V1 -> V2 ->V3. The repository has all the 4 variants that we tested.

Quick Start:
1) Install
cd LightGCN_work
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

2) Data layout

Place new datasets under LightGCN_work/data/<dataset_name>/.
Supported/used names in code: gowalla, amazon-book, instacart.

Expected files for models:

data/<dataset>/train.txt — lines: user item1 item2 ...

data/<dataset>/test.txt — same format (usually the last interaction(s) per user)

3) Prepare Instacart files into train.txt and test.txt

Convert raw Instacart CSVs → LightGCN text files:

# From LightGCN_work/
python code_V3_Fusion/prepare_instacart.py \
  --raw_dir ./data/instacart/raw \   (We haven't kept the raw folder in the git repo because of data size issue|)
  --out_dir ./data/instacart


Build an item–item adjacency (for item fusion):

python code_V3_Fusion/preprocess_instacart_i2i.py \
  --data_root ./data/instacart \
  --topk 20
# Result: ./data/instacart/i2i_adj.npz

Train & Evaluate
Minimal example (Gowalla)
# From LightGCN_work/ (for fusion variant V3)
python code_V3_Fusion/main.py \   
  --dataset gowalla --model lgn \
  --epochs 1000 --recdim 128 --layer 4 \
  --lr 0.0001 --bpr_batch 2048 

With extensions
# Popularity fusion + item–item fusion
python code_V3_Fusion/main.py \
  --dataset gowalla --model lgn \
  --epochs 1000 --recdim 128 --layer 4 \
  --use_item_item --i2i_path ./data/gowalla/i2i_adj.npz --i2i_alpha 0.1


Checkpoints → code/checkpoints/
CSV logs → code/checkpoints/*/train_epoch_metrics.csv and valid_epoch_metrics.csv
TensorBoard logs → code/runs/

Launch TensorBoard:

tensorboard --logdir=code/runs

What’s Inside
```
LightGCN_work/
├─ code_V3_Fusion/
│  ├─ main.py            # Training loop, resume, CSV/TensorBoard logging, Mac-safe multiprocessing guard
│  ├─ world.py           # Global config: args → config dict, paths, device selection
│  ├─ model.py           # LightGCN + popularity fusion (pop_mlp + gate_mlp) + optional item–item fusion
│  ├─ Procedure.py       # Train (BPR), eval (Precision/Recall/NDCG), CSV writers
│  ├─ parse.py           # CLI arguments (batch sizes, layers, pop-gate, item–item flags, etc.)
│  ├─ register.py        # Dataset loader & model registry
│  ├─ dataloader.py      # Dataset abstraction; sparse graph & splits
│  ├─ prepare_instacart.py          # Instacart → LightGCN train/test
│  ├─ preprocess_instacart_i2i.py   # Build item–item CSR (.npz)
│  └─ utils.py           # BPRLoss, samplers, metrics, misc helpers
└─ requirements.txt      # Includes tensorboardX==1.8
```
All others follow the same structure inside code_V1_Global_Smoothing and code_V2_MLP.


Key ideas for all Variants:

V0 — Baseline (LightGCN)

Bipartite user–item graph; L-hop neighborhood propagation with layer-wise averaging.

Final embedding = mean of all layer embeddings; score = user·item dot product.

Optimized with BPR loss; serves as the reproducible reference.

V1 — Global Smoothing

Adds a global smoothing term on top of LightGCN’s layer aggregation to reduce overfitting and amplify common structure.

Controlled by a single strength hyperparameter; leaves training/eval protocol unchanged so results are comparable to V0.

V2 — MLP Scoring (with optional residual blend)

Replaces the plain dot product with a small MLP that learns a richer interaction function.

Optionally blends scores: score = (1−α)·dot + α·MLP, where α is a simple scalar (e.g., --residual_alpha).

Same data splits/metrics as V0/V1 for apples-to-apples comparisons.

V3 - Fusion

LightGCN propagation: uniform neighbor “hops” for n_layers, then layer-mean aggregation.

Popularity fusion (Pop-Gate)-

Build a log-scaled, normalized popularity scalar per item.

Map it via pop_mlp to the embedding space.

Learn a gate sigmoid(gate_mlp ([item_emb, pop_vec])) to mix structural vs popularity signals.

Item–Item fusion: post-propagation, optionally smooth item embeddings with i2i_adj (CSR) weighted by i2i_alpha.


Common Flags-

Model / training: --model lgn, --epochs, --recdim, --layer, --lr, --decay

Batches: --bpr_batch

Logging: --tensorboard 1, --comment your_note

Resume: --resume, --resume_path path/to/ckpt.pth.tar

Item–item: --use_item_item --i2i_path ./data/<ds>/i2i_adj.npz --i2i_alpha 0.1

Run python code/main.py --help for the full list.

Notes:

If you enable TensorBoard (--tensorboard 1), runs appear in code/runs/.

If you use item–item fusion, ensure --i2i_path points to a valid .npz CSR.
