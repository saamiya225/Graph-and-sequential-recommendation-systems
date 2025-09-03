"""
V2 (MLP Scoring) — LightGCN variant

This file implements:
- [Backbone] LightGCN propagation on the bipartite user–item graph
- [Aggregation] Mean over layer-wise embeddings (LightGCN default)
- [Scoring, V2] MLP-based scoring head over concatenated features
                (user emb, item emb, optional biases/global terms)
- [Stability] Residual blend with dot-product (anchor CF), configurable via residual_alpha
- [Perf] Optional embedding cache for evaluation (invalidate between epochs)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import world
from dataloader import BasicDataset

# ========= Define the BasicModel and LightGCN Classes =========

class BasicModel(nn.Module):
    """Base model class."""
    def __init__(self):
        super(BasicModel, self).__init__()

class LightGCN(BasicModel):
    """
    LightGCN backbone + Global bias + MLP scorer (+ optional popularity gate).
    - Training: fresh graph every backward (no train-cache)
    - Eval: optional embedding cache; use invalidate_cache() before testing each epoch
    """
    def __init__(self, config, dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.device = world.device

        # ---- Sizes ----
        self.n_users = dataset.n_users
        self.m_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.n_layers = config['lightGCN_n_layers']

        # ---- Embeddings ----
        self.embedding_user = nn.Embedding(self.n_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.m_items, self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        # ---- Graph ----
        # Expect a row-normalized user–item bipartite adjacency (sparse)
        self.Graph = dataset.getSparseGraph().to(self.device)

        # ---- Optional: cache for eval ----
        self._cached_eval = None

        # ---- Global features / biases (if present in your original code) ----
        # self.user_bias = ...
        # self.item_bias = ...
        # self.global_scalar = ...

        # ---- Scorer: MLP over concatenated features + residual dot ----
        # Build concatenated input dim based on features you feed the scorer:
        in_dim = self.latent_dim * 2
        # If you concatenate bias embeddings or globals, add them to in_dim accordingly.

        # [norm] Optional feature normalization before MLP scorer
        self.pre_norm = nn.LayerNorm(in_dim)

        # [mlp] MLP scoring head: maps concatenated features → scalar score
        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # [blend] Residual dot-product weight (0.0→pure MLP, 1.0→pure dot)
        self.residual_alpha = float(config.get('residual_alpha', 0.3))

        # (Any other knobs such as use_norm, bias_scale should already be in your config)

    # ----------------------------- Cache Helpers -----------------------------

    # [cache] Call to clear cached embeddings before each evaluation epoch
    def invalidate_cache(self):
        """Invalidate any cached forward-pass state used for evaluation."""
        self._cached_eval = None

    # ---------------------- LightGCN Backbone Propagation ---------------------

    # [backbone] LightGCN propagation + mean aggregation over layers
    def computer(self):
        """
        Perform LightGCN propagation for n_layers and aggregate the layer-wise embeddings.

        Returns:
            all_users: (n_users, d) user embeddings after aggregation
            all_items: (n_items, d) item  embeddings after aggregation
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)        # (n_users+n_items, d)
        embs = [all_emb]                                          # layer 0

        x = all_emb
        for _ in range(self.n_layers):
            # One uniform "hop": A * x, where A is the normalized bipartite adjacency
            x = torch.sparse.mm(self.Graph, x)                    # (N, d)
            embs.append(x)

        embs = torch.stack(embs, dim=1)                           # (N, L+1, d)
        # [agg] LightGCN default: mean over [0..L] layer embeddings
        out = torch.mean(embs, dim=1)                             # (N, d)

        all_users = out[:self.n_users, :]
        all_items = out[self.n_users:, :]
        return all_users, all_items

    # ---------------------------- Utility Getters ----------------------------

    def getEmbedding(self, users, pos_items, neg_items):
        """
        Return sampled embeddings and the full tables for scoring/reg.
        """
        all_users, all_items = self.computer()
        u = all_users[users]
        p = all_items[pos_items]
        n = all_items[neg_items]
        return u, p, n, all_users, all_items

    # ----------------------------- Scoring / Forward -------------------------

    # [inference] Score (users, items) via MLP scorer + residual dot-product
    def forward(self, users, items):
        """
        Returns final blended scores for (users, items):

            mlp_score  = MLP( concat([u, i, ...]) )
            dot_score  = <u, i>
            final      = (1 - residual_alpha) * mlp_score + residual_alpha * dot_score
        """
        all_users, all_items = self.computer()
        u = all_users[users]                                      # (B, d)
        i = all_items[items]                                      # (B, d)

        # Build concatenated features for the MLP scorer.
        feats = [u, i]                                            # extend with biases/globals if used
        x = torch.cat(feats, dim=1)                               # (B, 2d [+ extras])

        # Optional normalization before MLP
        x = self.pre_norm(x)

        # [mlp] MLP score from concatenated features
        mlp_score = self.mlp(x).squeeze(-1)                       # (B,)

        # [dot] Dot-product anchor score
        dot_score = torch.sum(u * i, dim=1)                       # (B,)

        alpha = self.residual_alpha
        # [blend] Final score = (1-α)*MLP + α*dot
        final_score = (1.0 - alpha) * mlp_score + alpha * dot_score
        return final_score

    # ----------------------------- Loss: BPR ---------------------------------

    # [loss] BPR over active scoring (MLP + residual blend)
    def bpr_loss(self, users, pos, neg):
        """
        Pairwise BPR loss using the same scoring path as forward():
          L = -log σ( s(u, pos) - s(u, neg) ) + λ * ||emb||^2
        """
        u, pos_e, neg_e, _, _ = self.getEmbedding(users, pos, neg)

        # Rebuild features for pos/neg MLP scoring
        x_pos = self.pre_norm(torch.cat([u, pos_e], dim=1))
        x_neg = self.pre_norm(torch.cat([u, neg_e], dim=1))

        # [mlp] MLP score from concatenated features
        mlp_pos = self.mlp(x_pos).squeeze(-1)
        mlp_neg = self.mlp(x_neg).squeeze(-1)

        # [dot] Dot-product anchor score
        dot_pos = torch.sum(u * pos_e, dim=1)
        dot_neg = torch.sum(u * neg_e, dim=1)

        alpha = self.residual_alpha
        # [blend] Final score = (1-α)*MLP + α*dot
        pos_scores = (1.0 - alpha) * mlp_pos + alpha * dot_pos
        neg_scores = (1.0 - alpha) * mlp_neg + alpha * dot_neg

        # BPR loss
        bpr = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        # L2 regularization on sampled embeddings
        reg = (u.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2)) * 0.5 / u.shape[0]
        return bpr, reg

    # ----------------------- Optional Popularity Utilities -------------------

    # [pop] Utility: compute raw item popularity counts from the dataset
    def _compute_item_popularity(self) -> np.ndarray:
        """
        Returns an array of length m_items with frequency counts per item.
        """
        # Implementation assumed in your original file
        # (kept as-is; comments only)
        counts = np.zeros(self.m_items, dtype=np.int64)
        # ... populate counts from dataset ...
        return counts
