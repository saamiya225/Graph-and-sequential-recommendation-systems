import os
import csv
import world
import numpy as np
import torch
import utils
import dataloader
from utils import timer
import multiprocessing

CORES = multiprocessing.cpu_count() // 2

def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    # sampling
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
        cri = bpr.stageOne(u, p, n)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar('BPRLoss/BPR', cri,
                         epoch * total_batch + batch_i)
    aver_loss /= total_batch

    # ——— LOG TRAINING LOSS TO CSV ———
    save_path = world.config.get(
        'path',
        world.config.get('checkpoint_dir', './checkpoints')
    )
    os.makedirs(save_path, exist_ok=True)
    train_csv = os.path.join(save_path, 'train_epoch_metrics.csv')
    if not os.path.exists(train_csv):
        with open(train_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss'])
    with open(train_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, aver_loss])
    # ————————————————————————————————

    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"

def test_one_batch(X):
    sorted_items = X[0].cpu().numpy()
    groundTrue  = X[1]
    if not isinstance(groundTrue, (list, set, tuple, np.ndarray)):
        groundTrue = [groundTrue]
    # Now always batch-of-one
    test_data = [groundTrue]
    r = utils.getLabel(groundTrue, sorted_items)
    r = np.expand_dims(r, axis=0)  # [1, topk]

    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(test_data, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(test_data, r, k))
    return {'precision': np.array(pre),
            'recall':    np.array(recall),
            'ndcg':      np.array(ndcg)}




def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    testDict     = dataset.testDict
    Recmodel     = Recmodel.eval()
    max_K        = max(world.topks)
    if hasattr(Recmodel, "invalidate_cache"):
        Recmodel.invalidate_cache()
        
    pool = None
    
    if multicore == 1:
        # use world.CORES if your world.py defines it; else fall back to half CPUs
        try:
            CORES = world.CORES
        except AttributeError:
            CORES = multiprocessing.cpu_count() // 2 or 1
        pool = multiprocessing.Pool(CORES)

    results = {'precision': np.zeros(len(world.topks)),
               'recall':    np.zeros(len(world.topks)),
               'ndcg':      np.zeros(len(world.topks))}


    with torch.no_grad():
        users = list(testDict.keys())
        total_batch = len(users) // u_batch_size + 1
        batch_result = []
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos     = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            # Ensure every groundTruth is a list (or set/tuple), even if only one item
            groundTrue = [gt if isinstance(gt, (list, set, tuple)) else [gt] for gt in groundTrue]
            batch_gpu  = torch.Tensor(batch_users).long().to(world.device)

            rating_K = Recmodel.getUsersRating(batch_gpu)
            # mask seen interactions
            exclude_idx, exclude_items = [], []
            for i, items in enumerate(allPos):
                exclude_idx.extend([i] * len(items))
                exclude_items.extend(items)
            rating_K[exclude_idx, exclude_items] = -(1 << 10)

            _, topk = torch.topk(rating_K, k=max_K)
            # Loop over each user in the batch
            for i, u in enumerate(batch_users):
                X = (topk[i], groundTrue[i])
                batch_result.append(test_one_batch(X))

        # Aggregate results for each metric
        for metric in results.keys():
            results[metric] = np.mean([r[metric] for r in batch_result], axis=0)

    # ——— LOG VALIDATION METRICS TO CSV ———
    save_path = world.config.get(
        'path',
        world.config.get('checkpoint_dir', './checkpoints')
    )
    os.makedirs(save_path, exist_ok=True)
    valid_csv = os.path.join(save_path, 'valid_epoch_metrics.csv')
    if not os.path.exists(valid_csv):
        with open(valid_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'precision', 'recall', 'ndcg'])
    prec = float(results['precision'][0])
    rec  = float(results['recall'][0])
    nd   = float(results['ndcg'][0])
    with open(valid_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, prec, rec, nd])
    # ————————————————————————————————

    print(results)
    return results
