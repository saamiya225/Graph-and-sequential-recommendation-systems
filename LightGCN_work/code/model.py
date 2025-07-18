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
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
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
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
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
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is ready to go(dropout:{self.config['dropout']})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        return torch.sparse.FloatTensor(index.t(), values, size)
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = [self.__dropout_x(g, keep_prob) for g in self.Graph]
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        Graph propagation + global exponential smoothing over layers.
        Returns:
          users_final: Tensor [n_users, emb_dim]
          items_final: Tensor [n_items, emb_dim]
        """
        # 1) initial embeddings
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)

        embs = [all_emb]

        # 2) propagate for each layer
        if self.config['dropout'] and self.training:
            g = self.__dropout(self.keep_prob)
        else:
            g = self.Graph

        for _ in range(self.n_layers):
            if self.A_split:
                parts = [torch.sparse.mm(graph_part, all_emb) for graph_part in g]
                all_emb = torch.cat(parts, dim=0)
            else:
                all_emb = torch.sparse.mm(g, all_emb)
            embs.append(all_emb)

        # 3) stack embeddings: [n_nodes, n_layers+1, emb_dim]
        embs = torch.stack(embs, dim=1)

        # 4) global exponential smoothing weights
        beta = self.config.get('exp_smooth_beta', 0.5)
        K = self.n_layers
        device = embs.device
        weights = [(1 - beta) ** k for k in range(K + 1)]
        w = torch.tensor(weights, dtype=embs.dtype, device=device)
        w = w / w.sum()

        # 5) weighted sum over layers → [n_nodes, emb_dim]
        light_out = torch.sum(embs * w.view(1, K + 1, 1), dim=1)

        # 6) split back into users & items
        users_final, items_final = torch.split(
            light_out,
            [self.num_users, self.num_items],
            dim=0
        )
        return users_final, items_final

    def forward(self, users, items):
        """
        Given user & item index tensors, returns their dot-product scores.
        """
        users_emb, items_emb = self.computer()
        u_e = users_emb[users.long()]
        i_e = items_emb[items.long()]
        scores = torch.sum(u_e * i_e, dim=1)
        return scores
