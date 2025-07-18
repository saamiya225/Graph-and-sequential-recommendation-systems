"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError

    def forward(self, users, items):
        raise NotImplementedError

class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb * pos_emb, dim=1)
        neg_scores= torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (users_emb.norm(2).pow(2)
                    + pos_emb.norm(2).pow(2)
                    + neg_emb.norm(2).pow(2)) / (2.0 * len(users))
        return loss, reg_loss
    
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb * items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers   = self.config['lightGCN_n_layers']
        self.keep_prob  = self.config['keep_prob']
        self.A_split    = self.config['A_split']
        self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is ready to go(dropout:{self.config['dropout']})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        mask   = torch.rand(len(values)) + keep_prob
        mask   = mask.int().bool()
        index  = index[mask]
        values = values[mask] / keep_prob
        return torch.sparse_coo_tensor(index.t(), values, size)
    
    def __dropout(self, keep_prob):
        if self.A_split:
            return [self.__dropout_x(g, keep_prob) for g in self.Graph]
        return self.__dropout_x(self.Graph, keep_prob)

    def computer(self):
        # 1) initial embeddings
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb   = torch.cat([users_emb, items_emb], dim=0)
        embs      = [all_emb]

        # 2) propagate
        g = self.__dropout(self.keep_prob) if self.config['dropout'] and self.training else self.Graph
        for _ in range(self.n_layers):
            if self.A_split:
                parts   = [torch.sparse.mm(p, all_emb) for p in g]
                all_emb = torch.cat(parts, dim=0)
            else:
                all_emb = torch.sparse.mm(g, all_emb)
            embs.append(all_emb)

        # 3) stack: [n_nodes, K+1, dim]
        embs = torch.stack(embs, dim=1)

        # 4) exponential smoothing weights
        beta    = self.config.get('exp_smooth_beta', 0.5)
        K       = self.n_layers
        weights = [(1 - beta)**k for k in range(K+1)]
        w       = torch.tensor(weights, dtype=embs.dtype, device=embs.device)
        w       = w / w.sum()
        light_out = torch.sum(embs * w.view(1, K+1, 1), dim=1)

        # 5) split back
        users_final, items_final = torch.split(light_out, [self.num_users, self.num_items], dim=0)
        return users_final, items_final

    def getUsersRating(self, users):
        """
        Given a batch of user indices, returns a [batch, n_items] score matrix.
        """
        # 1) propagate once to get all user/item embeddings
        users_emb, items_emb = self.computer()     # users_emb: [n_users, D], items_emb: [n_items, D]
        # 2) select only the batch of users
        u_e = users_emb[users.long()]             # [batch, D]
        # 3) score = u_e @ items_emb^T → [batch, n_items]
        scores = torch.matmul(u_e, items_emb.t())
        return scores

    def bpr_loss(self, users, pos, neg):
        users_emb, items_emb = self.computer()
        u_emb = users_emb[users.long()]
        p_emb = items_emb[pos.long()]
        n_emb = items_emb[neg.long()]
        pos_scores = torch.sum(u_emb * p_emb, dim=1)
        neg_scores = torch.sum(u_emb * n_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (u_emb.norm(2).pow(2) + p_emb.norm(2).pow(2) + n_emb.norm(2).pow(2)) / (2.0 * len(users))
        return loss, reg_loss

    def forward(self, users, items):
        users_emb, items_emb = self.computer()
        u_e = users_emb[users.long()]
        i_e = items_emb[items.long()]
        return torch.sum(u_e * i_e, dim=1)
