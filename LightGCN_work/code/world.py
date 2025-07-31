import os
from parse import parse_args
from os.path import dirname, join
import sys

# workaround for Mac/KMP issue (if needed)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# parse all CLI args
args = parse_args()

# ——— Unified checkpoint/log path ———
# First try old `--path`, then `--checkpoint_dir`, else default
PATH = getattr(args, 'path', None)
if PATH is None:
    PATH = getattr(args, 'checkpoint_dir', './checkpoints')
# ————————————————————————————————

# project layout
ROOT_PATH  = dirname(dirname(__file__))
CODE_PATH  = join(ROOT_PATH, 'code')
DATA_PATH  = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH  = PATH    # used for saving weights & logs

# add any extra import paths
sys.path.append(join(CODE_PATH, 'sources'))

# build the global config dict
config = {
    'checkpoint_dir': FILE_PATH,
    'dataset':        args.dataset,
    'lr':             args.lr,
    'decay':          args.decay,
    'lightGCN_n_layers': args.layer,
    'latent_dim_rec': args.recdim,
    'bpr_batch_size': args.bpr_batch,
    'test_u_batch_size': args.testbatch,
    'dropout':        args.dropout,
    'keep_prob':      args.keepprob,
    'A_split':        bool(args.a_fold),
    'epochs':         args.epochs,
    'multicore':      args.multicore,
    'pretrain':       args.pretrain,
    'seed':           args.seed,
    'model':          args.model,
    'exp_smooth_beta': args.exp_smooth_beta,
    'use_ppr_weights': args.use_ppr_weights,
    'ppr_weights_path': args.ppr_weights_path,
    # … any other settings your code uses …
}

# device
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
