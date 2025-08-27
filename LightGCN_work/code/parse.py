"""
parse.py — Argument parser for LightGCN training.

This script defines all command-line arguments used across:
- world.py (global config setup)
- main.py (training loop)
- model.py (model architecture)

Typical usage:
    python main.py --dataset gowalla --epochs 500 --use_pop_gate
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go LightGCN")

    # === Core training parameters ===
    parser.add_argument('--bpr_batch',       type=int,   default=2048,
                        help='Batch size for BPR loss')
    parser.add_argument('--recdim',          type=int,   default=64,
                        help='Latent embedding dimension')
    parser.add_argument('--layer',           type=int,   default=3,
                        help='Number of LightGCN propagation layers')
    parser.add_argument('--lr',              type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--decay',           type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--dropout',         type=int,   default=0,
                        help='Graph dropout (0 = disabled)')
    parser.add_argument('--keepprob',        type=float, default=0.6,
                        help='Keep probability if dropout enabled')
    parser.add_argument('--epochs',          type=int,   default=1000,
                        help='Training epochs')
    parser.add_argument('--testbatch',       type=int,   default=100,
                        help='Batch size for testing')

    # === Dataset & paths ===
    parser.add_argument('--dataset',         type=str,   default='gowalla',
                        help='Dataset name')
    parser.add_argument('--checkpoint_dir',  type=str,   default='./checkpoints',
                        help='Path to save model checkpoints')
    parser.add_argument('--topks',           type=str,   default='[20]',
                        help='Evaluation top-K list, e.g. "[20, 40]"')

    # === Logging & reproducibility ===
    parser.add_argument('--tensorboard',     type=int,   default=1,
                        help='Enable TensorBoard logging (1=yes, 0=no)')
    parser.add_argument('--comment',         type=str,   default='lgn',
                        help='Experiment comment (used in TensorBoard run name)')
    parser.add_argument('--load',            type=int,   default=0,
                        help='Load pretrained model if available')
    parser.add_argument('--pretrain',        type=int,   default=0,
                        help='Use pretrained embeddings (if available)')
    parser.add_argument('--seed',            type=int,   default=2020,
                        help='Random seed')
    parser.add_argument('--model',           type=str,   default='lgn',
                        help='Model type (LightGCN variant)')
    parser.add_argument('--a_fold',          type=int,   default=100,
                        help='Adjacency fold for graph splitting')

    # Graph split toggle
    parser.add_argument('--A_split',         dest='A_split', action='store_true')
    parser.add_argument('--no-A_split',      dest='A_split', action='store_false')
    parser.set_defaults(A_split=False)

    # === Global smoothing / PPR (optional) ===
    parser.add_argument('--exp_smooth_beta', type=float, default=0.5,
                        help='Exponential smoothing β for edge weights')
    parser.add_argument('--use_ppr_weights', action='store_true',
                        help='Enable Personalized PageRank (PPR) weights')
    parser.add_argument('--ppr_weights_path', type=str, default=None,
                        help='Path to precomputed PPR weights')

    # === Scheduler (optional) ===
    parser.add_argument('--use_scheduler',   action='store_true',
                        help='Enable LR scheduler')
    parser.add_argument('--sched_milestones', type=str, default='[120,240,360,480]',
                        help='LR scheduler milestones (epochs)')
    parser.add_argument('--sched_gamma',    type=float, default=0.5,
                        help='LR decay factor at milestones')

    # === Popularity gate (pop fusion) ===
    parser.add_argument('--use_pop_gate',    action='store_true',
                        help='Enable popularity fusion + gate')
    parser.add_argument('--pop_hidden',      type=int, default=32,
                        help='Hidden units in pop MLP (scalar→hidden→recdim)')
    parser.add_argument('--gate_hidden',     type=int, default=64,
                        help='Hidden units in gate MLP')
    parser.add_argument('--gate_entropy_coeff', type=float, default=1e-4,
                        help='Coefficient for gate entropy regularization')
    parser.add_argument('--pop_gate_temp',   type=float, default=1.0,
                        help='Temperature for gate sigmoid (>1 = smoother gate)')

    # === Item–item adjacency (optional) ===
    parser.add_argument('--use_item_item',   action='store_true',
                        help='Enable item–item adjacency augmentation')
    parser.add_argument('--i2i_path',        type=str, default=None,
                        help='Path to item–item adjacency (.npz, CSR)')
    parser.add_argument('--i2i_alpha',       type=float, default=0.0,
                        help='Weight of item–item edge aggregation')

    # === Miscellaneous ===
    parser.add_argument('--multicore',       type=int,   default=0,
                        help='Use multiple cores in evaluation')
    parser.add_argument('--resume',          action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--resume_path',     type=str, default=None,
                        help='Path to specific checkpoint to resume from')
    parser.add_argument('--save_every',      type=int, default=10,
                        help='Save model every N epochs')

    return parser.parse_args()
