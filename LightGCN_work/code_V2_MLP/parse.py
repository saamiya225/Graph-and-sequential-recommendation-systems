"""
parse.py — V2 (MLP Scoring)


- [CLI] Core LightGCN arguments (dataset, epochs, layers, recdim, lr, etc.)
- [CLI, V2] MLP scorer is always enabled
    • --residual_alpha : blend weight between MLP score and dot-product
        (0.0 → pure MLP, 1.0 → pure dot-product, in-between → hybrid)
    • --use_norm       : L2-normalize embeddings before feeding MLP
    • --bias_scale     : scale factor for user/item bias embeddings
- [CLI] Other optional scheduler / logging / checkpoint args

"""

# (your original parse.py content below unchanged)
# Created on Mar 1, 2020
# Pytorch Implementation of LightGCN in
# Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
# ...
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")

   def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")

    # -------- Core training --------
    parser.add_argument('--bpr_batch',       type=int,   default=2048,
                        help="the batch size for BPR loss training")
    parser.add_argument('--recdim',          type=int,   default=64,
                        help="the embedding size of LightGCN")
    parser.add_argument('--layer',           type=int,   default=3,
                        help="number of GCN layers")
    parser.add_argument('--lr',              type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--decay',           type=float, default=1e-4,
                        help="weight decay for L2")
    parser.add_argument('--dropout',         type=int,   default=0,
                        help="whether to use adjacency dropout (0/1)")
    parser.add_argument('--keepprob',        type=float, default=0.6,
                        help="keep probability if dropout > 0")
    parser.add_argument('--epochs',          type=int,   default=1000,
                        help="number of training epochs")
    parser.add_argument('--testbatch',       type=int,   default=100,
                        help="user batch size for testing")

    # -------- Dataset / paths --------
    parser.add_argument('--dataset',         type=str,   default='gowalla',
                        help="dataset: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--checkpoint_dir',  type=str,   default='./checkpoints',
                        help="directory to save weights & logs")
    parser.add_argument('--topks',           nargs='?', default="[20]",
                        help="@k list for evaluation, e.g. \"[20]\"")
    parser.add_argument('--tensorboard',     type=int,   default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment',         type=str,   default="lgn",
                        help="run tag/comment")
    parser.add_argument('--load',            type=int,   default=0,
                        help="whether to load a pre-saved model (legacy)")
    parser.add_argument('--pretrain',        type=int,   default=0,
                        help="use pretrained embeddings")
    parser.add_argument('--seed',            type=int,   default=2020,
                        help="random seed")
    parser.add_argument('--model',           type=str,   default='lgn',
                        help="model: [mf, lgn]")

    # -------- Adjacency split (fixed boolean flags) --------
    parser.add_argument('--a_fold',          type=int,   default=100,
                        help="fold num to split large adj matrix (used if A_split)")
    parser.add_argument('--A_split', dest='A_split', action='store_true',
                        help="Split adjacency matrix into folds (use if OOM)")
    parser.add_argument('--no_A_split', dest='A_split', action='store_false',
                        help="Do NOT split adjacency (default)")
    parser.set_defaults(A_split=False)

    # -------- Global layer smoothing --------
    parser.add_argument('--exp_smooth_beta', type=float, default=0.5,
                        help="global exponential‐smoothing β for layer aggregation")

    # -------- MLP+Global scorer tuning (NEW) --------
    parser.add_argument('--residual_alpha',  type=float, default=0.0,
                        help='Blend dot-product with MLP score (0: MLP only, 1: dot only)')
    parser.add_argument('--use_norm',        action='store_true',
                        help='L2-normalize user/item embeddings before MLP scorer')
    parser.add_argument('--bias_scale',      type=float, default=1.0,
                        help='Scale factor for user/item bias embeddings into MLP')

    # -------- Multicore & resume & scheduler (not required as such, could use if needed, fir training other than the multicore, we didnt use others) --------
    parser.add_argument('--multicore',       type=int,   default=0,
                        help="use multiprocessing for test (0/1)")
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint in checkpoint_dir')
    parser.add_argument('--resume_path', type=str, default='',
                        help='Resume from a specific checkpoint file')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs (0=only last)')
    parser.add_argument('--keep_topk',  type=int, default=0,
                        help='Keep top-K best checkpoints by val NDCG (0=off)')
    parser.add_argument('--use_scheduler',   action='store_true',
                        help='Enable MultiStepLR with milestones for LR decay')
    parser.add_argument('--sched_milestones', type=str, default='[200, 300]',
                        help='Milestones list as string, e.g. [200, 300]')
    parser.add_argument('--sched_gamma',     type=float, default=0.5,
                        help='Decay factor for MultiStepLR')


    return parser.parse_args()

