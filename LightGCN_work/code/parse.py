"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
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
                        help="whether to use adjacency dropout")
    parser.add_argument('--keepprob',        type=float, default=0.6,
                        help="keep probability if dropout > 0")
    parser.add_argument('--a_fold',          type=int,   default=100,
                        help="fold num to split large adj matrix")
    parser.add_argument('--A_split', type=bool, default=False,
                    help="Whether to split adjacency matrix")
    parser.add_argument('--testbatch',       type=int,   default=100,
                        help="user batch size for testing")
    parser.add_argument('--dataset',         type=str,   default='gowalla',
                        help="dataset: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--checkpoint_dir',  type=str,   default='./checkpoints',
                        help="directory to save weights & logs")
    parser.add_argument('--topks',           nargs='?', default="[20]",
                        help="@k list for evaluation")
    parser.add_argument('--tensorboard',     type=int,   default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment',         type=str,   default="lgn",
                        help="run tag/comment")
    parser.add_argument('--load',            type=int,   default=0,
                        help="whether to load a pre-saved model")
    parser.add_argument('--epochs',          type=int,   default=1000,
                        help="number of training epochs")
    parser.add_argument('--multicore',       type=int,   default=0,
                        help="use multiprocessing for test")
    parser.add_argument('--pretrain',        type=int,   default=0,
                        help="use pretrained embeddings")
    parser.add_argument('--seed',            type=int,   default=2020,
                        help="random seed")
    parser.add_argument('--model',           type=str,   default='lgn',
                        help="model: [mf, lgn]")
    parser.add_argument('--exp_smooth_beta', type=float, default=0.5,
                        help="global exponential‐smoothing β")
    
    parser.add_argument('--use_ppr_weights', action='store_true', help='Use PPR weighting for layer combination')
    parser.add_argument('--ppr_weights_path', type=str, default=None, help='Path to PPR weights file (used if --use_ppr_weights is set)')

    return parser.parse_args()
