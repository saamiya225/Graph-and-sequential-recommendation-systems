"""
model.py — LightGCN with popularity fusion (pop-gate) and optional item-item graph.

This module defines:
- Utility function `_scipy_csr_to_torch_csr` to convert SciPy CSR → PyTorch CSR
- `LightGCN` class:
    - Standard LightGCN propagation over user–item bipartite graph
    - Embeddings for users/items
    - Optional popularity-aware gating mechanism ("pop-gate")
    - Optional item–item adjacency fusion
    - Loss function with BPR + gate entropy regularization
"""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp
import world


def _scipy_csr_to_torch_csr(sp_csr, device=None, dtype=torch.float32):
    """
    Convert SciPy CSR matrix into torch.sparse_csr_tensor.
    Ensures compatibility regardless of SciPy format variations.
    """
    if not sp.isspmatrix_csr(sp_csr):
        sp_csr = sp_csr.tocsr()
    indptr  = torch.from_numpy(sp_csr.indptr.astype(np.int64))
    indices = torch.from_numpy(sp_csr.indices.astype(np.int64))
    data    = torch.from_numpy(sp_csr.data.astype(np.float32))
    shape   = sp_csr.shape
    return torch.sparse_csr_tensor(indptr, indices, data, size=shape, dtype=dtype, device=device)


class LightGCN(nn.Module):
    def __init__(self, config, dataset):
        """
        Args:
            config  (dict): hyperparameters & flags
            dataset (object): dataset object with user/item counts & sparse graph
        """
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.device = world.device

        # Core sizes
        self.n_users = dataset.n_users
        self.m_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.n_layers = config['lightGCN_n_layers']
        self.keep_prob = config['keep_prob']

        # === Embedding layers ===
        self.embedding_user = nn.Embedding(self.n_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.m_items, self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        # Sparse user–item graph
        self.Graph = self.dataset.getSparseGraph()

        # === Popularity Gate ===
        self.use_pop_gate = bool(self.config.get('use_pop_gate', False))
        self.pop_hidden = int(self.config.get('pop_hidden', 32))
        self.gate_hidden = int(self.config.get('gate_hidden', 64))
        self.gate_entropy_coeff = float(self.config.get('gate_entropy_coeff', 1e-4))
        self.pop_gate_temp = float(self.config.get('pop_gate_temp', 1.0))

        if self.use_pop_gate:
            # Continuous popularity scalar per item
            counts = torch.from_numpy(self.dataset.items_D).float()
            counts = torch.clamp(counts, min=0.0)
            pop = torch.log1p(counts)                           # log scale
            pop = (pop - pop.mean()) / (pop.std() + 1e-8)       # normalize
            self.item_pop_scalar = pop.to(self.device)          # (m,)

            # Project scalar → latent dim
            self.pop_mlp = nn.Sequential(
                nn.Linear(1, self.pop_hidden),
                nn.ReLU(),
                nn.Linear(self.pop_hidden, self.latent_dim),
            )

            # Gate MLP: combines graph embedding & pop vector
            self.gate_mlp = nn.Sequential(
                nn.Linear(self.latent_dim * 2, self.gate_hidden),
                nn.ReLU(),
                nn.Linear(self.gate_hidden, 1)
            )
        else:
            self.item_pop_scalar = None
            self.pop_mlp = None
            self.gate_mlp = None

        # === Optional item–item graph ===
        self.use_item_item = bool(self.config.get('use_item_item', False))
        self.i2i_alpha     = float(self.config.get('i2i_alpha', 0.0))
        self.i2i_adj       = None
        if self.use_item_item and self.config.get('i2i_path', None):
            try:
                sp_i2i = sp.load_npz(self.config['i2i_path'])
                self.i2i_adj = _scipy_csr_to_torch_csr(sp_i2i, device=world.device, dtype=torch.float32)
                world.cprint(f"[I2I] loaded {self.config['i2i_path']}, nnz={sp_i2i.nnz}")
            except Exception as e:
                world.cprint(f"[I2I] WARNING: cannot load {self.config['i2i_path']}: {e}")
                self.i2i_adj = None

    # ----------------------------------------------------------------------
    # Embedding retrieval
    # ----------------------------------------------------------------------
    def getUsersRating(self, users):
        """
        Compute user–item scores for a batch of users.
        Applies pop-gate fusion if enabled.
        """
        all_users, all_items = self.computer()
        u_emb = all_users[users]   # (B, d)
        i_emb = self._fuse_item_embeddings(all_items) if self.use_pop_gate else all_items
        scores = torch.matmul(u_emb, i_emb.t())  # (B, M)
        return scores

    def getEmbedding(self, users, pos_items, neg_items):
        """
        Return tuple of embeddings for BPR training.
        """
        all_users, all_items = self.computer()
        i_emb = self._fuse_item_embeddings(all_items) if self.use_pop_gate else all_items
        u = all_users[users]
        pos = i_emb[pos_items]
        neg = i_emb[neg_items]
        return u, pos, neg, all_users, all_items

    # ----------------------------------------------------------------------
    # Pop gate fusion
    # ----------------------------------------------------------------------
    def _fuse_item_embeddings(self, items_emb):
        """
        Fuse LightGCN item embeddings with popularity vector.

        pop_vec = MLP(popularity scalar)
        gate    = sigmoid( gate_mlp([items_emb, pop_vec]) / T )
        fused   = gate * items_emb + (1 - gate) * pop_vec
        """
        pop_feat = self.item_pop_scalar.unsqueeze(1)    # (M,1)
        pop_vec  = self.pop_mlp(pop_feat)               # (M,d)
        gate_in  = torch.cat([items_emb, pop_vec], dim=1)
        gate_logit = self.gate_mlp(gate_in)
        if self.pop_gate_temp != 1.0:
            gate_logit = gate_logit / self.pop_gate_temp
        gate = torch.sigmoid(gate_logit)                # (M,1)

        fused = gate * items_emb + (1.0 - gate) * pop_vec
        self._last_item_gate = gate  # store for inspection/loss
        return fused

    # ----------------------------------------------------------------------
    # Loss
    # ----------------------------------------------------------------------
    def bpr_loss(self, users, pos, neg):
        """
        Bayesian Personalized Ranking (BPR) loss.
        Adds entropy regularization if pop-gate is enabled.
        """
        u, pos_e, neg_e, _, _ = self.getEmbedding(users, pos, neg)
        pos_scores = torch.sum(u * pos_e, dim=1)   # (B,)
        neg_scores = torch.sum(u * neg_e, dim=1)
        bpr = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # L2 regularization
        reg_loss = (0.5 * (u.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2))) / float(u.shape[0])

        loss = bpr
        if self.use_pop_gate and hasattr(self, "_last_item_gate"):
            # Entropy regularization (avoid gate collapse)
            gates = torch.cat([self._last_item_gate[pos], self._last_item_gate[neg]], dim=0)
            gates = torch.clamp(gates, 1e-6, 1.0-1e-6)
            entropy = -(gates * torch.log(gates) + (1-gates) * torch.log(1-gates)).mean()
            loss = loss - self.gate_entropy_coeff * entropy

        return loss, reg_loss

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(self, users, items):
        """
        Compute dot-product scores for given (users, items).
        """
        all_users, all_items = self.computer()
        i_emb = self._fuse_item_embeddings(all_items) if self.use_pop_gate else all_items
        u = all_users[users]
        i = i_emb[items]
        return (u * i).sum(dim=1)

    # ----------------------------------------------------------------------
    # Graph propagation
    # ----------------------------------------------------------------------
    def computer(self):
        """
        Perform LightGCN propagation over the user–item graph.
        Returns:
            all_users (n_users, d), all_items (m_items, d)
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        embs = [all_emb]

        g = self.Graph

        # TODO: implement edge dropout if required
        x = all_emb
        for _ in range(self.n_layers):
            x = torch.sparse.mm(g, x)
            embs.append(x)

        # Layer-wise aggregation (mean pooling)
        embs = torch.stack(embs, dim=1)  # (N, L+1, d)
        out = torch.mean(embs, dim=1)    # (N, d)

        all_users = out[:self.n_users, :]
        all_items = out[self.n_users:, :]

        # Optional item–item smoothing
        if self.use_item_item and (self.i2i_adj is not None) and (self.i2i_alpha > 0.0):
            all_items = all_items + self.i2i_alpha * torch.sparse.mm(self.i2i_adj, all_items)

        return all_users, all_items
