"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")

    # core
    parser.add_argument('--bpr_batch',       type=int,   default=2048, help="BPR batch size")
    parser.add_argument('--recdim',          type=int,   default=64,   help="embedding size")
    parser.add_argument('--layer',           type=int,   default=3,    help="num of GCN layers")
    parser.add_argument('--lr',              type=float, default=0.001,help="learning rate")
    parser.add_argument('--decay',           type=float, default=1e-4, help="L2 weight decay")
    parser.add_argument('--dropout',         type=int,   default=0,    help="use adj dropout (0/1)")
    parser.add_argument('--keepprob',        type=float, default=0.6,  help="keep prob if dropout > 0")
    parser.add_argument('--epochs',          type=int,   default=1000, help="training epochs")
    parser.add_argument('--testbatch',       type=int,   default=100,  help="user batch size for testing")
    parser.add_argument('--dataset',         type=str,   default='gowalla',
                        help="dataset: [lastfm, gowalla, yelp2018, amazon-book, instacart]")
    parser.add_argument('--checkpoint_dir',  type=str,   default='./checkpoints',
                        help="directory to save weights & logs")
    parser.add_argument('--topks',           nargs='?',  default="[20]",help="@k list for evaluation")
    parser.add_argument('--tensorboard',     type=int,   default=1,    help="enable tensorboard")
    parser.add_argument('--comment',         type=str,   default="lgn",help="run tag/comment")
    parser.add_argument('--load',            type=int,   default=0,    help="load pre-saved model (0/1)")
    parser.add_argument('--pretrain',        type=int,   default=0,    help="use pretrained embeddings")
    parser.add_argument('--seed',            type=int,   default=2020, help="random seed")
    parser.add_argument('--model',           type=str,   default='lgn',help="model: [mf, lgn]")

    # adjacency split (for very large graphs)
    parser.add_argument('--a_fold',          type=int,   default=100,  help="fold num to split adj")
    parser.add_argument('--A_split',         action=argparse.BooleanOptionalAction, default=False,
                        help="Whether to split adjacency matrix")

    # global smoothing / layer combination
    parser.add_argument('--exp_smooth_beta', type=float, default=0.5,  help="exp-smoothing β for layer weights")
    parser.add_argument('--use_ppr_weights', action='store_true',      help="(optional) PPR weighting")
    parser.add_argument('--ppr_weights_path', type=str, default=None,  help="path to PPR weights if used")

    # MLP / bias control
    parser.add_argument('--bias_scale',      type=float, default=1.0,  help="scale for user/item bias embeddings")

    # Popularity-gated fusion
    parser.add_argument('--use_pop_gate',    action='store_true',      help="Enable popularity-gated fusion")
    parser.add_argument('--pop_bins',        type=int,   default=10,   help="Number of popularity quantile bins")

    # Item–Item graph (Instacart only)
    parser.add_argument('--i2i_path',        type=str,   default=None, help="Path to Instacart i2i CSR .npz")

    # runtime
    parser.add_argument('--multicore',       type=int,   default=0,    help="use multiprocessing for test")
    parser.add_argument('--resume',          action='store_true',      help="resume training from a checkpoint")
    parser.add_argument('--resume_path',     type=str,   default=None, help="specific checkpoint path to resume from")
    parser.add_argument('--save_every',      type=int,   default=10,   help="save checkpoint every N epochs")

    # optional scheduler
    parser.add_argument('--use_scheduler',   action='store_true',      help="use MultiStepLR")
    parser.add_argument('--sched_milestones', type=str,  default="[120,240,360,480]",
                        help="milestones list, e.g. \"[120,240,360,480]\"")
    parser.add_argument('--sched_gamma',     type=float, default=0.5,  help="lr decay factor at milestones")

    # residual alpha for i2i fusion
    parser.add_argument('--residual_alpha',  type=float, default=0.2,  help="residual weight for i2i")

    return parser.parse_args()
