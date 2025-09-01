'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye
Updated: with comments for clarity
'''

import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
from model import LightGCN
from sklearn.metrics import roc_auc_score
import random
import os

# ==================== Negative Sampling Extension ====================
# Try to use the fast C++ sampler for BPR negative sampling.
# If unavailable, we fall back to the slower Python version.
try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except:
    world.cprint("Cpp extension not loaded")
    sample_ext = False


# ==================== BPR Loss ====================
class BPRLoss:
    """
    Bayesian Personalized Ranking (BPR) loss wrapper.

    - Handles forward pass through model’s bpr_loss()
    - Adds L2 regularization (weight decay)
    - Applies optimizer step
    """

    def __init__(self, recmodel: nn.Module, config: dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        # Forward pass: compute BPR + reg loss
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        # Backprop + update
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


# ==================== Sampling ====================
def UniformSample_original(dataset, neg_ratio=1):
    """
    Wrapper around sampling.
    If C++ extension is available, use it.
    Otherwise fall back to Python implementation.
    """
    dataset: BasicDataset
    allPos = dataset.allPos
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    return S


def UniformSample_original_python(dataset):
    """
    Python implementation of uniform negative sampling for BPR.
    For each user:
        - Pick one positive item
        - Pick one random negative item not in user’s history
    """
    dataset: BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []

    for user in users:
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        # sample one positive
        positem = np.random.choice(posForUser)
        # sample one negative (not in user’s positives)
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem not in posForUser:
                break
        S.append([user, positem, negitem])

    return np.array(S)


# ==================== Utility helpers ====================
def set_seed(seed):
    """Ensure reproducibility across NumPy, Torch, CUDA"""
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def getFileName():
    """
    Construct checkpoint filename depending on model.
    Example: lgn-amazon-book-3-128.pth.tar
    """
    if world.model_name == 'mf':
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'lgn':
        file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    return os.path.join(world.PATH, file)


def minibatch(*tensors, **kwargs):
    """Yield mini-batches from given tensors"""
    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])
    for i in range(0, len(tensors[0]), batch_size):
        yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    """Shuffle arrays in unison"""
    require_indices = kwargs.get('indices', False)
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError("All inputs must have same length.")

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)
    result = tuple(x[shuffle_indices] for x in arrays)
    return (result, shuffle_indices) if require_indices else result


# ==================== Timer ====================
class timer:
    """
    Lightweight timing utility for profiling.
    Usage:
        with timer():
            do_something()
        print(timer.get())
    """
    from time import time
    TAPE = [-1]
    NAMED_TAPE = {}

    @staticmethod
    def get():
        return timer.TAPE.pop() if len(timer.TAPE) > 1 else -1


# ==================== Evaluation Metrics ====================
def RecallPrecision_ATk(test_data, r, k):
    """
    Compute recall and precision at top-k.
    test_data: ground-truth items per user
    r: binary relevance matrix (batch x items)
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain at top-k.
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        test_matrix[i, :min(k, len(items))] = 1
    idcg = np.sum(test_matrix * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = np.sum(pred_data * (1./np.log2(np.arange(2, k + 2))), axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    return np.sum(ndcg)


def AUC(all_item_scores, dataset, test_data):
    """
    Area Under Curve: given all item scores for one user.
    """
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    return roc_auc_score(r_all, all_item_scores)


def getLabel(groundTruth, predictTopK):
    """Return binary labels for predicted top-K vs ground-truth set"""
    if not isinstance(groundTruth, (list, set, tuple, np.ndarray)):
        groundTruth = [groundTruth]
    pred = [1.0 if x in groundTruth else 0.0 for x in predictTopK]
    return np.array(pred, dtype=np.float32)
