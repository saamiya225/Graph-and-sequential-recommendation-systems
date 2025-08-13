# main.py — LightGCN (+ Global+MLP) with resume, scheduler, and FULL CSV logging
# (No AMP / No extra speed-up flags; evaluates every 10 epochs)

import os
import time
import ast
import csv
import torch
import numpy as np

import world
import utils
from world import cprint
from tensorboardX import SummaryWriter
from os.path import join

import register
from register import dataset
import Procedure

# ============================== helpers ==============================

def _ckpt_dir():
    d = world.FILE_PATH if hasattr(world, "FILE_PATH") else world.config['checkpoint_dir']
    os.makedirs(d, exist_ok=True)
    return d

def _last_ckpt_path():
    return os.path.join(_ckpt_dir(), 'last.pth.tar')

def _best_ckpt_path(epoch):
    return os.path.join(_ckpt_dir(), f'best-epoch{epoch}.pth.tar')

def _legacy_tag_path():
    # legacy single-file path used by older scripts
    return utils.getFileName()

def save_checkpoint(model, optimizer, epoch, best_metric=None, scheduler=None, path=None):
    path = path or _last_ckpt_path()
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        'best_metric': float(best_metric) if best_metric is not None else None,
    }
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, device=None):
    device = device or world.device
    ckpt = torch.load(path, map_location=device)
    # support both rich checkpoints and raw state_dict
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'], strict=True)
        if optimizer is not None and ckpt.get('optimizer_state') is not None:
            try: optimizer.load_state_dict(ckpt['optimizer_state'])
            except Exception: pass
        if scheduler is not None and ckpt.get('scheduler_state') is not None:
            try: scheduler.load_state_dict(ckpt['scheduler_state'])
            except Exception: pass
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        best_metric = ckpt.get('best_metric', None)
    else:
        # raw state_dict
        model.load_state_dict(ckpt, strict=True)
        start_epoch = 1
        best_metric = None
    return start_epoch, best_metric

def ndcg_at_topk(results_dict):
    if not results_dict or 'ndcg' not in results_dict:
        return None
    v = results_dict['ndcg']
    if isinstance(v, (list, tuple, np.ndarray)):
        return float(v[0]) if len(v) else None
    return float(v)

# -------- CSV logging helpers --------
def _csv_paths():
    d = _ckpt_dir()
    return os.path.join(d, "train_epoch_metrics.csv"), os.path.join(d, "valid_epoch_metrics.csv")

def _ensure_csv_headers(train_csv, valid_csv, topks):
    # Train CSV
    if not os.path.exists(train_csv):
        with open(train_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "time_sec", "train_loss", "lr"])
            writer.writeheader()
    # Valid CSV with dynamic topk columns
    if not os.path.exists(valid_csv):
        fields = ["epoch", "time_sec", "lr"]
        for k in topks:
            fields += [f"precision@{k}", f"recall@{k}", f"ndcg@{k}"]
        with open(valid_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

def _append_train_csv(train_csv, epoch, time_sec, train_loss, lr):
    with open(train_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "time_sec", "train_loss", "lr"])
        writer.writerow({
            "epoch": epoch,
            "time_sec": f"{time_sec:.3f}",
            "train_loss": f"{train_loss:.6f}",
            "lr": f"{lr:.8f}"
        })

def _append_valid_csv(valid_csv, epoch, time_sec, lr, metrics, topks):
    row = {"epoch": epoch, "time_sec": f"{time_sec:.3f}", "lr": f"{lr:.8f}"}
    for i, k in enumerate(topks):
        row[f"precision@{k}"] = f"{float(metrics['precision'][i]):.8f}" if 'precision' in metrics else ""
        row[f"recall@{k}"]    = f"{float(metrics['recall'][i]):.8f}"    if 'recall' in metrics else ""
        row[f"ndcg@{k}"]      = f"{float(metrics['ndcg'][i]):.8f}"      if 'ndcg' in metrics else ""
    fields = ["epoch", "time_sec", "lr"] + \
             [f"precision@{k}" for k in topks] + \
             [f"recall@{k}" for k in topks] + \
             [f"ndcg@{k}" for k in topks]
    if not os.path.exists(valid_csv):
        with open(valid_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
    else:
        with open(valid_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(row)

# ============================== seed & setup ==============================

utils.set_seed(world.seed)
print(">>SEED:", world.seed)

# ============================== build model & loss ==============================

Recmodel = register.MODELS[world.model_name](world.config, dataset).to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)   # exposes .opt

optimizer = bpr.opt
scheduler = None
if world.config.get('use_scheduler', False):
    milestones = ast.literal_eval(world.config.get('sched_milestones', '[200, 300]'))
    gamma = float(world.config.get('sched_gamma', 0.5))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    cprint(f"[Scheduler] MultiStepLR milestones={milestones} gamma={gamma}")

# ============================== resume logic ==============================

weight_file = _legacy_tag_path()      # legacy single-file
last_ckpt   = _last_ckpt_path()

print(f"load and save to {weight_file}")
start_epoch = 1
best_ndcg = None

# legacy path if world.LOAD==1
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=world.device))
        cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

# new resume flow (preferred)
resume_flag = world.config.get('resume', False)
resume_path = world.config.get('resume_path', '')
if resume_flag or resume_path:
    candidate = None
    if resume_path and os.path.exists(resume_path):
        candidate = resume_path
    elif os.path.exists(last_ckpt):
        candidate = last_ckpt
    elif os.path.exists(weight_file):
        candidate = weight_file
    if candidate:
        try:
            start_epoch, best_ndcg = load_checkpoint(candidate, Recmodel, optimizer, scheduler, world.device)
            cprint(f"[RESUME] loaded '{candidate}' -> start_epoch={start_epoch}, best_ndcg={best_ndcg}")
        except Exception as e:
            cprint(f"[RESUME] failed to load '{candidate}': {e}")

Neg_k = 1

# ============================== tensorboard ==============================

if world.tensorboard:
    w: SummaryWriter = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment))
else:
    w = None
    cprint("not enable tensorflowboard")

# ============================== CSV set-up ==============================

train_csv, valid_csv = _csv_paths()
_ensure_csv_headers(train_csv, valid_csv, world.topks)

# ============================== training loop ==============================

SAVE_EVERY = int(world.config.get('save_every', 5))
KEEP_TOPK  = int(world.config.get('keep_topk', 0))
best_paths = []

try:
    for epoch in range(start_epoch, world.TRAIN_epochs + 1):
        # ---- TEST every 10 epochs (like original) ----
        if (epoch - 1) % 10 == 0:
            cprint("[TEST]")
            Recmodel.eval()
            # IMPORTANT: avoid stale eval embeddings
            if hasattr(Recmodel, "invalidate_cache"):
                Recmodel.invalidate_cache()
            t0 = time.time()
            ret = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            t_eval = time.time() - t0
            cprint(f"[TEST] {ret}")

            # CSV: full metrics
            current_lr = optimizer.param_groups[0]['lr']
            _append_valid_csv(valid_csv, epoch, t_eval, current_lr, ret, world.topks)

            ndcg_now = ndcg_at_topk(ret)
            # Save best-by-ndcg (does not affect CSV logging)
            if ndcg_now is not None:
                is_better = (best_ndcg is None) or (ndcg_now > best_ndcg)
                if is_better:
                    best_ndcg = ndcg_now
                    best_path = _best_ckpt_path(epoch)
                    save_checkpoint(Recmodel, optimizer, epoch, best_metric=best_ndcg, scheduler=scheduler, path=best_path)
                    best_paths.append(best_path)
                    cprint(f"[BEST] epoch {epoch} new best NDCG={best_ndcg:.6f} -> {best_path}")
                    # trim to top-K
                    if KEEP_TOPK and len(best_paths) > KEEP_TOPK:
                        old = best_paths.pop(0)
                        try: os.remove(old)
                        except OSError: pass

        # ---- TRAIN ----
        t0 = time.time()
        S = utils.UniformSample_original(dataset)
        users = torch.tensor(S[:, 0], dtype=torch.long, device=world.device)
        pos   = torch.tensor(S[:, 1], dtype=torch.long, device=world.device)
        neg   = torch.tensor(S[:, 2], dtype=torch.long, device=world.device)
        users, pos, neg = utils.shuffle(users, pos, neg)

        ep_loss = 0.0
        num_steps = 0
        for (batch_users, batch_pos, batch_neg) in utils.minibatch(users, pos, neg, batch_size=world.config['bpr_batch_size']):
            loss = bpr.stageOne(batch_users, batch_pos, batch_neg)
            ep_loss += loss
            num_steps += 1

        # step scheduler
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        avg_loss = ep_loss / max(1, num_steps)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'EPOCH[{epoch}/{world.TRAIN_epochs}] loss{avg_loss:.3f}-|Sample+Train:{elapsed:.2f}|')

        # CSV: train log
        _append_train_csv(train_csv, epoch, elapsed, avg_loss, current_lr)

        # ---- SAVE last and periodic legacy ----
        save_checkpoint(Recmodel, optimizer, epoch, best_metric=best_ndcg, scheduler=scheduler, path=_last_ckpt_path())

        if SAVE_EVERY and (epoch % SAVE_EVERY == 0):
            # also write legacy single-file state_dict for compatibility
            try:
                torch.save(Recmodel.state_dict(), weight_file)
            except Exception:
                pass

finally:
    if world.tensorboard and w is not None:
        w.close()
