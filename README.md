**Limitations of User-Item Interactions in Graph Recommenders.**

Lightweight experiments in graph-based recommendation centered on LightGCN, with practical extensions:

Popularity fusion (Pop-Gate) — blends structural item embeddings with popularity-aware embeddings via a learned gate.

Item–Item adjacency fusion — optional smoothing using a precomputed item–item graph.

(Config stubs present for global smoothing / PPR; current code averages layers uniformly unless extended.)

The codebase includes training, evaluation, preprocessing (Instacart), checkpointing, and logging (CSV + TensorBoard).

Model evolution:

We started with the base code of LightGCN https://github.com/gusye1234/LightGCN-PyTorch.git, and implemented different versions to test our research questions. The final codebase uploaded is the last version in a sries of modifications. We went from V1 -> V2 ->V3.

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
python code/prepare_instacart.py \
  --raw_dir ./data/instacart/raw \   (We haven't kept the raw folder in the git repo because of data size issue|)
  --out_dir ./data/instacart


Build an item–item adjacency (for item fusion):

python code/preprocess_instacart_i2i.py \
  --data_root ./data/instacart \
  --topk 50 --weight jaccard
# Result: ./data/instacart/i2i_adj.npz

Train & Evaluate
Minimal example (Gowalla)
# From LightGCN_work/
python code/main.py \
  --dataset gowalla --model lgn \
  --epochs 200 --recdim 64 --layer 3 \
  --lr 0.001 --bpr_batch 2048 --testbatch 100

With extensions
# Popularity fusion + item–item fusion
python code/main.py \
  --dataset gowalla --model lgn \
  --epochs 1000 --recdim 128 --layer 3 \
  --use_item_item --i2i_path ./data/gowalla/i2i_adj.npz --i2i_alpha 0.1


Checkpoints → code/checkpoints/
CSV logs → code/checkpoints/*/train_epoch_metrics.csv and valid_epoch_metrics.csv
TensorBoard logs → code/runs/

Launch TensorBoard:

tensorboard --logdir=code/runs

What’s Inside
LightGCN_work/
├─ code/
│  ├─ main.py            # Entry point: training loop, resume, CSV/TensorBoard logging, Mac-safe multiprocessing guard
│  ├─ world.py           # Global config: args → config dict, paths, device selection
│  ├─ model.py           # LightGCN + popularity fusion (pop_mlp + gate_mlp) + optional item–item fusion
│  ├─ Procedure.py       # Epoch train (BPR), eval (Precision/Recall/NDCG), CSV writers
│  ├─ parse.py           # CLI arguments (batch sizes, layers, pop-gate flags, item–item flags, etc.)
│  ├─ register.py        # Dataset loader & model registry; sanity checks
│  ├─ dataloader.py      # Dataset abstraction; provides sparse graph & splits
│  ├─ prepare_instacart.py          # Instacart → LightGCN train/test
│  ├─ preprocess_instacart_i2i.py   # Build item–item CSR (.npz)
│  └─ utils.py           # BPRLoss, samplers, metrics, misc helpers
└─ requirements.txt      # Includes tensorboardX==1.8 (for logging)

Key ideas:

LightGCN propagation: uniform neighbor “hops” for n_layers, then layer-mean aggregation.

Popularity fusion (Pop-Gate)-

Build a log-scaled, normalized popularity scalar per item.

Map it via pop_mlp to the embedding space.

Learn a gate sigmoid(gate_mlp([item_emb, pop_vec])) to mix structural vs popularity signals.

Item–Item fusion: post-propagation, optionally smooth item embeddings with i2i_adj (CSR) weighted by i2i_alpha.

Global smoothing / PPR- flags exist in config/args. Current code averages layers uniformly unless you add weighting logic. The PPR model did not work well, and so it was scrapped from the final version.

Common Flags-

Model / training: --model lgn, --epochs, --recdim, --layer, --lr, --decay

Batches: --bpr_batch, --test_u_batch_size

Logging: --tensorboard 1, --comment your_note

Resume: --resume, --resume_path path/to/ckpt.pth.tar

Item–item: --use_item_item --i2i_path ./data/<ds>/i2i_adj.npz --i2i_alpha 0.1

Run python code/main.py --help for the full list.

Notes:

If you enable TensorBoard (--tensorboard 1), runs appear in code/runs/.

If you use item–item fusion, ensure --i2i_path points to a valid .npz CSR.
