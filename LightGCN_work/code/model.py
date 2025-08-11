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
    def __init__(self, config: dict, dataset: BasicDataset):
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
        users_emb = self.embedding_user(users.long())
        items_emb = self.embedding_item.weight
        scores    = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        u_emb = self.embedding_user(users.long())
        p_emb = self.embedding_item(pos.long())
        n_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(u_emb * p_emb, dim=1)
        neg_scores = torch.sum(u_emb * n_emb, dim=1)
        loss       = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg        = (u_emb.norm(2).pow(2)
                    + p_emb.norm(2).pow(2)
                    + n_emb.norm(2).pow(2)) / (2.0 * len(users))
        return loss, reg

    def forward(self, users, items):
        u_emb = self.embedding_user(users.long())
        i_emb = self.embedding_item(items.long())
        scores = torch.sum(u_emb * i_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config  = config
        self.dataset = dataset
        self.device  = world.device
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
        if self.config.get('pretrain', 0) == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))

        # Robust MLP gate expects 2 features (norm, mean) → (K+1) hops
        self.layer_gate = nn.Linear(2, self.n_layers + 1)
        nn.init.xavier_uniform_(self.layer_gate.weight)
        nn.init.zeros_(self.layer_gate.bias)

        self.Graph = self.dataset.getSparseGraph()
        if isinstance(self.Graph, torch.Tensor):
            self.Graph = self.Graph.coalesce().to(self.device)
        else:
            self.Graph = [g.coalesce().to(self.device) for g in self.Graph]

        self.f = nn.Sigmoid()
        world.cprint(f"LightGCN+MLP ready (dropout={self.config['dropout']})")

    def __dropout_x(self, x, keep_prob):
        idx  = x.indices().t()
        vals = x.values()
        mask = (torch.rand(len(vals), device=self.device) + keep_prob).int().bool()
        idx  = idx[mask]
        vals = vals[mask] / keep_prob
        return torch.sparse_coo_tensor(idx.t(), vals, x.size(), device=self.device)

    def __dropout(self, keep_prob):
        if self.A_split:
            return [self.__dropout_x(g, keep_prob) for g in self.Graph]
        return self.__dropout_x(self.Graph, keep_prob)

    def computer(self):
        users_emb = self.embedding_user.weight                # [n_users, d]
        items_emb = self.embedding_item.weight                # [n_items, d]
        all_emb   = torch.cat([users_emb, items_emb], dim=0) # [n_nodes, d]
        embs      = [all_emb]

        g = self.__dropout(self.keep_prob) if (self.config['dropout'] and self.training) else self.Graph
        for _ in range(self.n_layers):
            if self.A_split:
                parts   = [torch.sparse.mm(part, all_emb) for part in g]
                all_emb = torch.cat(parts, dim=0)
            else:
                all_emb = torch.sparse.mm(g, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)  # [n_nodes, K+1, d]

        # More expressive features for gating
        raw   = embs[:, 0]                                    # [n_nodes, d]
        feat_norm = raw.norm(dim=1, keepdim=True)             # [n_nodes, 1]
        feat_mean = raw.mean(dim=1, keepdim=True)             # [n_nodes, 1]
        feat = torch.cat([feat_norm, feat_mean], dim=1)       # [n_nodes, 2]

        logits = self.layer_gate(feat)                        # [n_nodes, K+1]
        alpha = torch.softmax(logits, dim=1)                  # [n_nodes, K+1]

        # Uncomment the print below to debug gate values (rarely prints)
        # if self.training and torch.rand(1).item() < 0.01:
        #     print('alpha min:', alpha.min().item(), 'max:', alpha.max().item(), 'mean:', alpha.mean().item())

        light_out = torch.sum(embs * alpha.unsqueeze(2), dim=1)  # [n_nodes, d]

        users_final, items_final = torch.split(
            light_out,
            [self.num_users, self.num_items],
            dim=0
        )
        return users_final, items_final

    def getUsersRating(self, users):
        users_emb, items_emb = self.computer()
        u_e = users_emb[users.long()]
        return self.f(torch.matmul(u_e, items_emb.t()))

    def bpr_loss(self, users, pos, neg):
        users_emb, items_emb = self.computer()
        u = users_emb[users.long()]
        p = items_emb[pos.long()]
        n = items_emb[neg.long()]
        pos_scores = torch.sum(u * p, dim=1)
        neg_scores = torch.sum(u * n, dim=1)
        loss       = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg        = (u.norm(2).pow(2)
                    + p.norm(2).pow(2)
                    + n.norm(2).pow(2)) / (2.0 * len(users))
        return loss, reg

    def forward(self, users, items):
        users_emb, items_emb = self.computer()
        u_e = users_emb[users.long()]
        i_e = items_emb[items.long()]
        return torch.sum(u_e * i_e, dim=1)
