# world.py — central config + globals expected across the repo

import os, sys, ast, multiprocessing
from os.path import dirname, join
from parse import parse_args

# Workaround for MKL/KMP clashes (esp. on Mac)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# -------- parse CLI --------
args = parse_args()

# -------- simple print wrapper (some files import this) --------
def cprint(*a, **k):
    print(*a, **k)

# -------- project paths --------
ROOT_PATH  = dirname(dirname(__file__))
CODE_PATH  = join(ROOT_PATH, 'code')
DATA_PATH  = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')

# prefer --checkpoint_dir; keep original behavior naming as FILE_PATH
FILE_PATH  = args.checkpoint_dir
os.makedirs(FILE_PATH, exist_ok=True)

# allow other modules to import sources if needed
sys.path.append(join(CODE_PATH, 'sources'))

# -------- expose frequently used args as module-level variables --------
dataset       = args.dataset
model_name    = args.model
tensorboard   = args.tensorboard
LOAD          = args.load
TRAIN_epochs  = args.epochs
seed          = args.seed
comment       = args.comment
PATH       = FILE_PATH  

# tops-K list
try:
    topks = ast.literal_eval(args.topks)
except Exception:
    topks = [20]

# number of CPU cores some code prints/uses
CORES = max(1, multiprocessing.cpu_count() // 2)

# -------- build config dict (what models/dataloader expect) --------
# A_split can come either from explicit --A_split or legacy behaviour (bool(a_fold))
A_split_flag = getattr(args, 'A_split', None)
if A_split_flag is None:
    A_split_flag = bool(args.a_fold)

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
    'A_split':           A_split_flag,
    'A_n_fold':          args.a_fold,     # <— note: was args.A_n_fold in your error; keep lowercase here
    'epochs':            args.epochs,
    'multicore':         args.multicore,
    'pretrain':          args.pretrain,
    'seed':              args.seed,
    'model':             args.model,
    'exp_smooth_beta':   getattr(args, 'exp_smooth_beta', 0.5),
    'use_ppr_weights':   getattr(args, 'use_ppr_weights', False),
    'ppr_weights_path':  getattr(args, 'ppr_weights_path', None),

    # optional knobs some versions expect (safe defaults if unused)
    'residual_alpha':    getattr(args, 'residual_alpha', 0.0),
    'use_norm':          getattr(args, 'use_norm', False),
    'bias_scale':        getattr(args, 'bias_scale', 1.0),
    'use_pop_gate':      getattr(args, 'use_pop_gate', False),
    'pop_bins':          getattr(args, 'pop_bins', 5),

    # checkpointing / scheduler (safe defaults)
    'resume':            getattr(args, 'resume', False),
    'resume_path':       getattr(args, 'resume_path', ''),
    'save_every':        getattr(args, 'save_every', 5),
    'keep_topk':         getattr(args, 'keep_topk', 0),
    'use_scheduler':     getattr(args, 'use_scheduler', False),
    'sched_milestones':  getattr(args, 'sched_milestones', '[200, 300]'),
    'sched_gamma':       getattr(args, 'sched_gamma', 0.5),
}

# -------- device --------
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
