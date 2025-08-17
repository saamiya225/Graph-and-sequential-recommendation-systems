import os
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import world

############################
# Helpers
############################

def _scipy_csr_to_torch_csr(sp_csr, device, dtype=torch.float32):
    indptr  = torch.from_numpy(sp_csr.indptr.astype(np.int64))
    indices = torch.from_numpy(sp_csr.indices.astype(np.int64))
    data    = torch.from_numpy(sp_csr.data).to(dtype)
    return torch.sparse_csr_tensor(indptr, indices, data, size=sp_csr.shape, device=device)

def _normalize_layer_weights(n_layers: int, beta: float = 0.0):
    """
    Return per-layer weights for combining LightGCN layer outputs.
    If beta > 0: exponential smoothing weight ~ beta^l (l=0..L).
    Otherwise uniform.
    """
    if beta > 0:
        weights = np.array([beta ** l for l in range(n_layers + 1)], dtype=np.float64)
        weights = weights / (weights.sum() + 1e-12)
        return torch.tensor(weights, dtype=torch.float32)
    else:
        return torch.ones(n_layers + 1, dtype=torch.float32) / float(n_layers + 1)

############################
# LightGCN + Two-Tower MLP
############################

class LightGCN(nn.Module):
    """
    LightGCN backbone with:
      - Two-tower MLP scoring: score(u,i) = < f_u(e_u), f_i(e_i^fused) > + b_u + b_i
      - Popularity-gated item fusion: e_i^g = (1 - z_i)*e_i + z_i*p_i
      - Optional Item–Item graph fusion (Instacart): e_i^final = e_i^g + alpha * (A_i2i @ e_i)
    """
    def __init__(self, config, dataset):
        super().__init__()
        self.config  = config
        self.dataset = dataset

        self.n_users = dataset.n_users
        self.n_items = dataset.m_items

        self.latent_dim  = int(config['latent_dim_rec'])
        self.n_layers    = int(config['lightGCN_n_layers'])
        self.keep_prob   = float(config['keep_prob']) if int(config['dropout']) else 1.0
        self.A_split     = bool(config['A_split'])
        self.exp_beta    = float(config.get('exp_smooth_beta', 0.0))

        # embeddings
        self.user_emb = nn.Embedding(self.n_users, self.latent_dim)
        self.item_emb = nn.Embedding(self.n_items, self.latent_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        # biases
        self.bias_scale = float(config.get('bias_scale', 1.0))
        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items, 1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        # two-tower MLPs
        # user tower: R^d -> R^d
        self.user_tower = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        # item tower: R^d -> R^d (applied after fusion)
        self.item_tower = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        # popularity-gated fusion
        self.use_pop_gate = bool(config.get('use_pop_gate', False))
        self.pop_bins     = int(config.get('pop_bins', 10))
        self.pop_emb      = nn.Embedding(self.pop_bins, self.latent_dim)
        nn.init.normal_(self.pop_emb.weight, std=0.01)
        self.gate_mlp     = nn.Sequential(
            nn.Linear(2 * self.latent_dim, self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim, 1)
        )

        # i2i message passing (Instacart only)
        self.i2i_adj = None
        self.residual_alpha = float(config.get('residual_alpha', 0.2))
        if world.dataset == 'instacart':
            i2i_path = config.get('i2i_path', None)
            if i2i_path is None:
                i2i_path = os.path.join(world.DATA_PATH, 'instacart', 'i2i_adj.npz')
            if os.path.exists(i2i_path):
                from scipy.sparse import load_npz
                sp_i2i = load_npz(i2i_path)
                self.i2i_adj = _scipy_csr_to_torch_csr(sp_i2i, device=world.device, dtype=torch.float32)
                world.cprint(f"[I2I] Loaded item-item CSR from {i2i_path}. nnz={sp_i2i.nnz}")
            else:
                world.cprint(f"[I2I] WARNING: not found at {i2i_path}. Continuing without i2i.")

        # precompute pop-bins from training degree
        self._build_item_pop_bins()

        # layer weights for LightGCN combination
        self.register_buffer('layer_weights', _normalize_layer_weights(self.n_layers, beta=self.exp_beta))

        # adjacency from dataset
        self.Graph = self.dataset.getSparseGraph()  # torch sparse (coalesced) OR list of folds

        # NOTE: We intentionally do not cache per-epoch embeddings to ensure fresh graph each backward.
        # (As requested).

    def _build_item_pop_bins(self):
        """
        Compute item popularity bins from training counts or item degree.
        Registers: self.item_pop_bins (LongTensor, [n_items])
        """
        try:
            item_freq = getattr(self.dataset, 'items_D', None)
        except Exception:
            item_freq = None
        if item_freq is None:
            # fallback: count from trainItem if available
            if hasattr(self.dataset, 'trainItem'):
                freq = np.zeros(self.n_items, dtype=np.int64)
                for it in self.dataset.trainItem:
                    if it < self.n_items:
                        freq[int(it)] += 1
                item_freq = freq
            else:
                item_freq = np.ones(self.n_items, dtype=np.int64)

        ranks = np.argsort(np.argsort(item_freq))
        denom = max(1, len(item_freq) - 1)
        quant = ranks.astype(np.float64) / float(denom)  # in [0,1]
        bin_ids = np.floor(quant * self.pop_bins).astype(np.int64)
        bin_ids[bin_ids == self.pop_bins] = self.pop_bins - 1

        self.register_buffer('item_pop_bins', torch.from_numpy(bin_ids))

    def getUsersRating(self, users: torch.Tensor) -> torch.Tensor:
        """
        users: LongTensor [B]
        return: scores for all items: [B, n_items]
        """
        self.train(False)
        # fresh embeddings from graph
        all_users, all_items = self.computer()
        # fuse item tower
        all_items_fused = self._fuse_item_tower(all_items)

        # towers
        U = self.user_tower(all_users[users])          # [B, d]
        I = self.item_tower(all_items_fused)           # [N, d]

        # scores = U @ I^T + biases
        scores = torch.matmul(U, I.transpose(0,1))     # [B, N]
        # add bias
        u_b = self.user_bias(users).view(-1, 1) * self.bias_scale
        i_b = self.item_bias.weight.view(1, -1) * self.bias_scale
        scores = scores + u_b + i_b
        return scores

    def computer(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LightGCN propagation: sum layer outputs with weights.
        Returns: final user, item embeddings after combination.
        """
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)

        embs = [all_emb]
        g = self.Graph
        if isinstance(g, list):
            # split graph
            for _ in range(self.n_layers):
                temp_emb = []
                for G in g:
                    side_emb = torch.sparse.mm(G, embs[-1])
                    temp_emb.append(side_emb)
                embs.append(torch.cat(temp_emb, dim=0))
        else:
            # single graph
            for _ in range(self.n_layers):
                all_emb = torch.sparse.mm(g, embs[-1])
                embs.append(all_emb)

        # weighted sum of layers
        layer_ws = self.layer_weights.to(embs[0].device)  # [L+1]
        # stack: [(n, d)] -> (L+1, n, d)
        stack = torch.stack(embs, dim=0)
        # weighted sum across 0th dim
        out = (layer_ws.view(-1, 1, 1) * stack).sum(dim=0)

        # split back
        users, items = torch.split(out, [self.n_users, self.n_items])
        return users, items

    def _i2i_message(self, E_item: torch.Tensor) -> torch.Tensor:
        """
        Item–Item message passing (Instacart).
        E_item: [n_items, d] -> return same shape
        """
        if self.i2i_adj is None:
            return torch.zeros_like(E_item)
        return torch.spmm(self.i2i_adj, E_item)  # CSR @ dense

    def _pop_gate_item(self, E_item: torch.Tensor) -> torch.Tensor:
        """
        Popularity-gated fusion:
          p_i = pop_emb[bin(i)]
          z_i = sigmoid(MLP([E_i || p_i]))
          E_i^g = (1-z_i)*E_i + z_i*p_i
        """
        if not self.use_pop_gate:
            return E_item
        pop_ids = self.item_pop_bins.to(E_item.device)
        p_vec   = self.pop_emb(pop_ids)           # [n_items, d]
        gate_in = torch.cat([E_item, p_vec], dim=-1)
        z       = torch.sigmoid(self.gate_mlp(gate_in))  # [n_items, 1]
        return (1.0 - z) * E_item + z * p_vec

    def _fuse_item_tower(self, E_item: torch.Tensor) -> torch.Tensor:
        """
        Apply pop-gate, then optional i2i residual: E_i^final = E_i^g + alpha * (A_i2i @ E_i)
        """
        Ei = self._pop_gate_item(E_item)
        if self.i2i_adj is not None:
            Ii = self._i2i_message(E_item)
            Ei = Ei + self.residual_alpha * Ii
        return Ei

    def bpr_loss(self, users, pos, neg):
        """
        Standard BPR loss with two-tower scoring.
        users, pos, neg: LongTensors of same length
        """
        self.train(True)
        # fresh embeddings
        all_users, all_items = self.computer()
        all_items_fused = self._fuse_item_tower(all_items)

        u_e = all_users[users]           # [B, d]
        p_e = all_items_fused[pos]       # [B, d]
        n_e = all_items_fused[neg]       # [B, d]

        u_b = self.user_bias(users) * self.bias_scale  # [B, 1]
        p_b = self.item_bias(pos)   * self.bias_scale  # [B, 1]
        n_b = self.item_bias(neg)   * self.bias_scale  # [B, 1]

        u_z = self.user_tower(u_e)   # [B, d]
        p_z = self.item_tower(p_e)   # [B, d]
        n_z = self.item_tower(n_e)   # [B, d]

        pos_scores = (u_z * p_z).sum(dim=-1, keepdim=True) + u_b + p_b  # [B,1]
        neg_scores = (u_z * n_z).sum(dim=-1, keepdim=True) + u_b + n_b  # [B,1]

        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # L2 regularization on base embeddings
        reg = (u_e.norm(2).pow(2) + p_e.norm(2).pow(2) + n_e.norm(2).pow(2)) / users.shape[0]
        return loss, reg

    def forward(self, users, items=None):
        """
        If items is None: return scores over all items for given users.
        Else return pairwise scores for the given (users, items).
        """
        if items is None:
            return self.getUsersRating(users)

        # else: pairwise scores
        all_users, all_items = self.computer()
        all_items_fused = self._fuse_item_tower(all_items)

        u_e = all_users[users]             # [B, d]
        i_e = all_items_fused[items]       # [B, d]

        u_z = self.user_tower(u_e)         # [B, d]
        i_z = self.item_tower(i_e)         # [B, d]

        u_b = self.user_bias(users) * self.bias_scale  # [B, 1]
        i_b = self.item_bias(items) * self.bias_scale  # [B, 1]

        scores = (u_z * i_z).sum(dim=-1, keepdim=True) + u_b + i_b
        return scores.squeeze(-1)

    @torch.no_grad()
    def invalidate_cache(self):
        """
        Kept for API compatibility; we do not cache per-epoch tensors.
        """
        return
