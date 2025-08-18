import argparse
import ast

def parse_args():
    parser = argparse.ArgumentParser()

    # ---------- Core model ----------
    parser.add_argument('--model', type=str, default='lgn')
    parser.add_argument('--dataset', type=str, default='amazon-book')
    parser.add_argument('--recdim', type=int, default=64)
    parser.add_argument('--layer', type=int, default=3)

    # ---------- Training ----------
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--bpr_batch', type=int, default=2048)
    parser.add_argument('--testbatch', type=int, default=100)
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--keepprob', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=2020)

    # ---------- Checkpoints ----------
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--pretrain', type=int, default=0)

    # ---------- Evaluation ----------
    parser.add_argument('--topks', type=str, default='[20]')
    parser.add_argument('--tensorboard', type=int, default=1)
    parser.add_argument('--comment', type=str, default='')

    # ---------- Graph processing ----------
    parser.add_argument('--a_fold', type=int, default=100)
    parser.add_argument('--A_split', dest='A_split', action='store_true')
    parser.add_argument('--no-A_split', dest='A_split', action='store_false')
    parser.set_defaults(A_split=False)

    # ---------- Scheduler ----------
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--sched_milestones', type=str, default='[120,240,360,480]')
    parser.add_argument('--sched_gamma', type=float, default=0.5)

    # ---------- Popularity gating ----------
    parser.add_argument('--use_pop_gate', action='store_true')
    parser.add_argument('--pop_bins', type=int, default=10)
    parser.add_argument('--pop_embed_dim', type=int, default=16)
    parser.add_argument('--bias_scale', type=float, default=1.0)

    # ---------- Fusion MLP ----------
    parser.add_argument('--use_factor_mlp', action='store_true')
    parser.add_argument('--proj_hidden', type=int, default=64)

    # ---------- Item-Item Graph ----------
    parser.add_argument('--use_item_item', action='store_true')
    parser.add_argument('--i2i_path', type=str, default=None)
    parser.add_argument('--i2i_alpha', type=float, default=0.2)

    # ---------- Extra smoothing & residual ----------
    parser.add_argument('--exp_smooth_beta', type=float, default=0.0)
    parser.add_argument('--residual_alpha', type=float, default=0.0)

    # ---------- PPR weights ----------
    parser.add_argument('--use_ppr_weights', action='store_true')
    parser.add_argument('--ppr_weights_path', type=str, default=None)

    # ---------- Multiprocessing ----------
    parser.add_argument('--multicore', type=int, default=0)

    args = parser.parse_args()

    # convert string args to python objects
    args.topks = ast.literal_eval(args.topks)
    if isinstance(args.sched_milestones, str):
        args.sched_milestones = ast.literal_eval(args.sched_milestones)

    return args
