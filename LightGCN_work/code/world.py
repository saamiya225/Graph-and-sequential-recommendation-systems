import os, sys, ast, multiprocessing
from os.path import dirname, join
from parse import parse_args

# Avoid MKL duplicate on some setups
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

args = parse_args()
CORES = multiprocessing.cpu_count() // 2
topks = ast.literal_eval(args.topks)

# Paths
ROOT_PATH  = dirname(dirname(__file__))
CODE_PATH  = join(ROOT_PATH, 'code')
DATA_PATH  = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH  = args.checkpoint_dir
sys.path.append(join(CODE_PATH, 'sources'))

config = {
    'checkpoint_dir':    args.checkpoint_dir,
    'dataset':           args.dataset,
    'lr':                args.lr,
    'decay':             args.decay,
    'lightGCN_n_layers': args.layer,
    'latent_dim_rec':    args.recdim,
    'bpr_batch_size':    args.bpr_batch,
    'test_u_batch_size': args.testbatch,
    'dropout':           args.dropout,
    'keep_prob':         args.keepprob,
    'A_split':           args.A_split,
    'A_n_fold':          args.A_n_fold,
    'epochs':            args.epochs,
    'multicore':         args.multicore,
    'pretrain':          args.pretrain,
    'seed':              args.seed,
    'model':             args.model,
    'exp_smooth_beta':   args.exp_smooth_beta,
    'use_ppr_weights':   args.use_ppr_weights,
    'ppr_weights_path':  args.ppr_weights_path,

    # Scorer tuning (NEW)
    'residual_alpha':    args.residual_alpha,
    'use_norm':          args.use_norm,
    'bias_scale':        args.bias_scale,

    # Popularity gate (optional)
    'use_pop_gate':      args.use_pop_gate,
    'pop_bins':          args.pop_bins,

    # Resume
    'resume':            args.resume,
    'resume_path':       args.resume_path,
    'save_every':        args.save_every,
    'keep_topk':         args.keep_topk,

    # Scheduler
    'use_scheduler':     args.use_scheduler,
    'sched_milestones':  args.sched_milestones,
    'sched_gamma':       args.sched_gamma,
}

# Device
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
