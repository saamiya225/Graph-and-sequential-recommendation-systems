"""
world.py — Global configuration and environment setup for training/evaluation.

This script:
- Parses command-line arguments
- Sets up environment variables and paths
- Defines hardware resources (CPU cores, CUDA device)
- Prepares a central `config` dictionary containing all hyperparameters
  and options used throughout the project

Inputs: Command-line arguments (see parse.py for details)
Outputs: Global variables & `config` dict used by other modules
"""

import os
import ast
import torch
import multiprocessing
from os.path import dirname, join
from parse import parse_args

# Prevents duplicate library errors with some BLAS backends
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Parse CLI arguments
args = parse_args()

# Simple print wrapper (could be extended for colored/conditional logging)
def cprint(*args_, **kwargs): 
    print(*args_, **kwargs)

# Core parameters from args
seed        = args.seed
dataset     = args.dataset
comment     = args.comment
tensorboard = args.tensorboard
LOAD        = args.load
model_name  = args.model
TRAIN_epochs= args.epochs
# Parse top-k list if provided as a string, else use directly
topks       = ast.literal_eval(args.topks) if isinstance(args.topks, str) else args.topks

# CPU cores: default to half available, fallback = 4
try:
    CORES = multiprocessing.cpu_count() // 2
except:
    CORES = 4

# Define project paths
ROOT_PATH  = dirname(dirname(__file__))
CODE_PATH  = join(ROOT_PATH, 'code')
DATA_PATH  = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
PATH       = args.checkpoint_dir

# Central configuration dictionary (passed around to other modules)
config = {
    'checkpoint_dir':     PATH,
    'dataset':            args.dataset,
    'lr':                 args.lr,
    'decay':              args.decay,
    'lightGCN_n_layers':  args.layer,
    'latent_dim_rec':     args.recdim,
    'bpr_batch_size':     args.bpr_batch,
    'test_u_batch_size':  args.testbatch,
    'dropout':            args.dropout,
    'keep_prob':          args.keepprob,
    'A_split':            args.A_split,
    'A_n_fold':           args.a_fold,
    'epochs':             args.epochs,
    'multicore':          args.multicore,
    'pretrain':           args.pretrain,
    'seed':               args.seed,
    'model':              args.model,

    # global smoothing / personalized PageRank (optional)
    'exp_smooth_beta':    args.exp_smooth_beta,
    'use_ppr_weights':    args.use_ppr_weights,
    'ppr_weights_path':   args.ppr_weights_path,

    # scheduler
    'use_scheduler':      args.use_scheduler,
    'sched_gamma':        args.sched_gamma,
}

# Scheduler milestones: fallback if args are missing/badly formatted
try:
    config['sched_milestones'] = (
        list(map(int, ast.literal_eval(args.sched_milestones)))
        if isinstance(args.sched_milestones, str) 
        else list(args.sched_milestones)
    )
except Exception:
    config['sched_milestones'] = [120, 240, 360, 480]

# Popularity gate parameters
config['use_pop_gate']       = args.use_pop_gate
config['pop_hidden']         = args.pop_hidden
config['gate_hidden']        = args.gate_hidden
config['gate_entropy_coeff'] = args.gate_entropy_coeff
config['pop_gate_temp']      = args.pop_gate_temp

# Item-item recommendation augmentation
config['use_item_item']      = args.use_item_item
config['i2i_path']           = args.i2i_path
config['i2i_alpha']          = args.i2i_alpha

# Select device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
