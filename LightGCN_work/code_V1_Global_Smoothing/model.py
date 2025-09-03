"""
V1 (Global Smoothing) — LightGCN

What this variant implements:
- [Backbone] LightGCN propagation on the user–item bipartite graph (uniform neighbor hopping)
- [Aggregation] Mean over layer-wise embeddings (LightGCN default)
- [Scoring] Dot-product scoring 〈u, i〉
- [Smoothing Hooks] Optional config for PPR/global smoothing (layer weights), if you wire it in
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import world
from dataloader import BasicDataset

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

class LightGCN(BasicModel):
    def __init__(self, config, dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.device = world.device

        self.n_users = dataset.n_users
        self.m_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.n_layers = config['lightGCN_n_layers']

        # Embeddings
        self.embedding_user = nn.Embedding(self.n_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.m_items, self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        # Graph adjacency (row-normalized user–item bipartite graph)
        self.Graph = dataset.getSparseGraph().to(self.device)

    # [cache] Refresh any cached embeddings before each evaluation epoch
    def invalidate_cache(self):
        pass

    # [backbone] LightGCN propagation + layer aggregation
    def computer(self):
        """
        Propagate embeddings across the bipartite graph and
        aggregate layer-wise representations.
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        embs = [all_emb]

        x = all_emb
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.Graph, x)
            embs.append(x)

        embs = torch.stack(embs, dim=1)
        # [agg] LightGCN default: mean over layer embeddings (0..L)
        out = torch.mean(embs, dim=1)

        all_users = out[:self.n_users, :]
        all_items = out[self.n_users:, :]
        return all_users, all_items

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        u = all_users[users]
        p = all_items[pos_items]
        n = all_items[neg_items]
        return u, p, n, all_users, all_items

    def forward(self, users, items):
        all_users, all_items = self.computer()
        u = all_users[users]
        i = all_items[items]
        # [score] Dot-product scoring 〈u, i〉
        return (u * i).sum(dim=1)

    def bpr_loss(self, users, pos, neg):
        u, pos_e, neg_e, _, _ = self.getEmbedding(users, pos, neg)
        # [bpr] Positive scores via dot product
        pos_scores = torch.sum(u * pos_e, dim=1)
        # [bpr] Negative scores via dot product
        neg_scores = torch.sum(u * neg_e, dim=1)

        bpr = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        reg = (u.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2)) * 0.5 / u.shape[0]
        return bpr, reg
