"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
"""

import argparse

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

    # -------- Optional PPR weighting --------
    parser.add_argument('--use_ppr_weights', action='store_true',
                        help='Use PPR weighting for layer combination')
    parser.add_argument('--ppr_weights_path', type=str, default=None,
                        help='Path to PPR weights file (used if --use_ppr_weights is set)')

    # -------- MLP+Global scorer tuning (NEW) --------
    parser.add_argument('--residual_alpha',  type=float, default=0.0,
                        help='Blend dot-product with MLP score (0: MLP only, 1: dot only)')
    parser.add_argument('--use_norm',        action='store_true',
                        help='L2-normalize user/item embeddings before MLP scorer')
    parser.add_argument('--bias_scale',      type=float, default=1.0,
                        help='Scale factor for user/item bias embeddings into MLP')

    # -------- Popularity-gated fusion (OFF by default) --------
    parser.add_argument('--use_pop_gate', action='store_true',
                        help='Enable popularity-gated item fusion')
    parser.add_argument('--pop_bins',     type=int, default=5,
                        help='Quantile bins for item popularity when gating is enabled')

    # -------- Multicore & resume & scheduler (QoL) --------
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
