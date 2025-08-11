import os
from parse import parse_args
from os.path import dirname, join
import sys
import multiprocessing
import ast

# workaround for Mac/KMP issue (if needed)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# parse all CLI args
args = parse_args()
seed = args.seed
dataset = args.dataset
comment = args.comment
tensorboard = args.tensorboard
LOAD = args.load
model_name = args.model
TRAIN_epochs = args.epochs
# number of CPU cores to use when multicore=1
CORES = multiprocessing.cpu_count() // 2
topks = ast.literal_eval(args.topks)


def cprint(*args, **kwargs):
    # simple wrapper around print, kept for backwards compatibility
    print(*args, **kwargs)
    
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
    'checkpoint_dir':    FILE_PATH,
    'dataset':           args.dataset,
    'lr':                args.lr,
    'decay':             args.decay,
    'lightGCN_n_layers': args.layer,
    'latent_dim_rec':    args.recdim,
    'bpr_batch_size':    args.bpr_batch,
    'test_u_batch_size': args.testbatch,
    'dropout':           args.dropout,
    'keep_prob':         args.keepprob,
    'A_split':           False,
    'A_n_fold':          args.a_fold,         
    'epochs':            args.epochs,
    'multicore':         args.multicore,
    'pretrain':          args.pretrain,
    'seed':              args.seed,
    'model':             args.model,
    'exp_smooth_beta':   args.exp_smooth_beta,
    'use_ppr_weights':   args.use_ppr_weights,
    'ppr_weights_path':  args.ppr_weights_path,
    # …any other settings…
}


# device
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
