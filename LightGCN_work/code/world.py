import os
import ast
import torch
import multiprocessing
from os.path import dirname, join
from parse import parse_args

# workaround for Mac/KMP issue (if needed)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

args = parse_args()

def cprint(*args_, **kwargs):
    print(*args_, **kwargs)

# Basic globals
seed         = args.seed
dataset      = args.dataset
comment      = args.comment
tensorboard  = args.tensorboard
LOAD         = args.load
model_name   = args.model
TRAIN_epochs = args.epochs

# topks may be a string like "[20,50]"
topks = ast.literal_eval(args.topks) if isinstance(args.topks, str) else args.topks

# CORES
try:
    CORES = multiprocessing.cpu_count() // 2
except Exception:
    CORES = 4

# Paths
ROOT_PATH  = dirname(dirname(__file__))
CODE_PATH  = join(ROOT_PATH, 'code')
DATA_PATH  = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
PATH       = args.checkpoint_dir or './checkpoints'

# Build config dict
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
    # prefer explicit A_split if provided, else fallback to bool(a_fold)
    'A_split':            args.A_split if hasattr(args, 'A_split') else bool(args.a_fold),
    'A_n_fold':           args.a_fold,
    'epochs':             args.epochs,
    'multicore':          args.multicore,
    'pretrain':           args.pretrain,
    'seed':               args.seed,
    'model':              args.model,

    # Layer weighting / smoothing / PPR
    'exp_smooth_beta':    args.exp_smooth_beta,
    'use_ppr_weights':    args.use_ppr_weights,
    'ppr_weights_path':   args.ppr_weights_path,

    # pop gate + factorized MLP
    'use_pop_gate':       args.use_pop_gate,
    'pop_embed_dim':      args.pop_embed_dim,
    'use_factor_mlp':     args.use_factor_mlp,
    'proj_hidden':        args.proj_hidden,

    # item-item
    'use_item_item':      args.use_item_item,
    'i2i_alpha':          args.i2i_alpha,
    'i2i_path':           args.i2i_path,

    # resume/scheduler
    'resume':             args.resume,
    'resume_path':        args.resume_path,
    'use_scheduler':      args.use_scheduler,
    'sched_gamma':        args.sched_gamma,
}

# ---- Normalize pop_bins to int (robust) ----
# Accepts: 10, "10", or even "[10]" -> 10
try:
    if isinstance(args.pop_bins, int):
        config['pop_bins'] = args.pop_bins
    elif isinstance(args.pop_bins, str):
        val = ast.literal_eval(args.pop_bins)
        if isinstance(val, (list, tuple)) and len(val) > 0:
            config['pop_bins'] = int(val[0])
        else:
            config['pop_bins'] = int(val)
    else:
        config['pop_bins'] = 10
except Exception:
    config['pop_bins'] = 10

# ---- Parse scheduler milestones as a list[int] ----
# Accepts: [120,240] or " [120, 240] "
try:
    if isinstance(args.sched_milestones, str):
        parsed = ast.literal_eval(args.sched_milestones)
        if isinstance(parsed, (list, tuple)):
            config['sched_milestones'] = [int(x) for x in parsed]
        else:
            config['sched_milestones'] = [120, 240, 360, 480]
    elif isinstance(args.sched_milestones, (list, tuple)):
        config['sched_milestones'] = [int(x) for x in args.sched_milestones]
    else:
        config['sched_milestones'] = [120, 240, 360, 480]
except Exception:
    config['sched_milestones'] = [120, 240, 360, 480]

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
