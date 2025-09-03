"""
Procedure.py — V1 (Global Smoothing)

- [train] One BPR epoch: sample (user, pos, neg), compute BPR + L2, step optimizer
- [eval] Precision/Recall/NDCG@K with masking of seen items
- [log] Write train/valid metrics to CSV (and TensorBoard if enabled)

"""

import os
import csv
import world
import numpy as np
import torch
import utils
from utils import timer
import multiprocessing

CORES = multiprocessing.cpu_count() // 2

# [train] One epoch of BPR training with dot-product scoring (V1)
def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    # [sample] Uniformly sample (user, pos, neg) triplets for BPR
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users    = torch.Tensor(S[:, 0]).long().to(world.device)
    posItems = torch.Tensor(S[:, 1]).long().to(world.device)
    negItems = torch.Tensor(S[:, 2]).long().to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)

    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.0
    for batch_i, (u, p, n) in enumerate(utils.minibatch(
            users, posItems, negItems,
            batch_size=world.config['bpr_batch_size'])):
        # [step] Forward + backward on one minibatch → returns current BPR loss
        cri = bpr.stageOne(u, p, n)
        aver_loss += cri
        if world.tensorboard and w is not None:
            w.add_scalar('BPRLoss/BPR', cri, epoch * total_batch + batch_i)
    aver_loss /= total_batch

    # Save training loss to CSV
    save_path = world.config.get('checkpoint_dir', './checkpoints')
    os.makedirs(save_path, exist_ok=True)
    train_csv = os.path.join(save_path, 'train_epoch_metrics.csv')
    if not os.path.exists(train_csv):
        with open(train_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'loss'])
    with open(train_csv, 'a', newline='') as f:
        csv.writer(f).writerow([epoch, aver_loss])

    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"

# [eval] Full evaluation: top-K per user + Precision/Recall/NDCG@K
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    from world import CORES
    u_batch_size = world.config['test_u_batch_size']
    testDict     = dataset.testDict
    Recmodel     = Recmodel.eval()
    max_K        = max(world.topks)

    results = {'precision': np.zeros(len(world.topks)),
               'recall':    np.zeros(len(world.topks)),
               'ndcg':      np.zeros(len(world.topks))}

    with torch.no_grad():
        users = list(testDict.keys())
        batch_result = []
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos     = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_gpu  = torch.Tensor(batch_users).long().to(world.device)

            rating_K = Recmodel.getUsersRating(batch_gpu)
            # [mask] Exclude items already interacted with by the user
            exclude_idx, exclude_items = [], []
            for i, items in enumerate(allPos):
                exclude_idx.extend([i] * len(items))
                exclude_items.extend(items)
            rating_K[exclude_idx, exclude_items] = -(1 << 10)

            _, topk = torch.topk(rating_K, k=max_K)
            for i, u in enumerate(batch_users):
                X = (topk[i], groundTrue[i])
                r = utils.test_one_batch(X)
                batch_result.append(r)

        # [aggregate] Average metric across evaluated users
        for metric in results.keys():
            results[metric] = np.mean([r[metric] for r in batch_result], axis=0)

    # Save validation metrics to CSV
    valid_csv = os.path.join(world.config.get('checkpoint_dir', './checkpoints'), 'valid_epoch_metrics.csv')
    if not os.path.exists(valid_csv):
        with open(valid_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'precision', 'recall', 'ndcg'])
    prec, rec, nd = results['precision'][0], results['recall'][0], results['ndcg'][0]
    with open(valid_csv, 'a', newline='') as f:
        csv.writer(f).writerow([epoch, prec, rec, nd])

    print(results)
    return results
