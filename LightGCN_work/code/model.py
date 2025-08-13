import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import world
from dataloader import BasicDataset

class BasicModel(nn.Module):
    def __init__(self): super().__init__()

class LightGCN(BasicModel):
    """
    LightGCN backbone + Global bias + MLP scorer (+ optional popularity gate).
    - Training: fresh graph every backward (no train-cache)
    - Eval: optional embedding cache; use invalidate_cache() before testing each epoch
    """
    def __init__(self, config, dataset):
        super().__init__()
        self.config  = config
        self.dataset = dataset
        self.device  = world.device

        self.latent_dim = int(config.get('latent_dim_rec', 128))
        self.n_layers   = int(config.get('lightGCN_n_layers', 4))

        # Embeddings
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        # Bias embeddings
        self.user_bias = nn.Embedding(self.num_users, self.latent_dim)
        self.item_bias = nn.Embedding(self.num_items, self.latent_dim)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        self.bias_scale = float(config.get('bias_scale', 0.5))

        # Graph
        self.Graph = dataset.getSparseGraph().to(self.device)

        # Global + MLP scorer
        in_dim = self.latent_dim * 5  # u, i, u_bias, i_bias, global
        self.pre_norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Residual dot-product blend (anchor CF)
        self.residual_alpha = float(config.get('residual_alpha', 0.3))
        self.use_norm = bool(config.get('use_norm', False))

        # Popularity-Gated Fusion (off unless enabled)
        self.use_pop_gate = bool(config.get('use_pop_gate', False))
        self.pop_bins     = int(config.get('pop_bins', 5))
        if self.use_pop_gate:
            counts = self._compute_item_popularity()
            bins, _ = self._bin_popularity(counts, self.pop_bins)
            self.register_buffer('item_pop_bin', torch.from_numpy(bins).long().to(self.device))
            self.pop_emb = nn.Embedding(self.pop_bins, self.latent_dim)
            self.gate_i = nn.Linear(self.latent_dim, self.latent_dim, bias=True)
            self.gate_p = nn.Linear(self.latent_dim, self.latent_dim, bias=False)

        # Eval-time cache (optional)
        self._cached_users = None
        self._cached_items = None

    # ---------- Graph propagation ----------
    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb   = torch.cat([users_emb, items_emb], dim=0)
        embs = [all_emb]
        g = self.Graph
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g, all_emb)
            embs.append(all_emb)
        all_emb = torch.mean(torch.stack(embs, dim=1), dim=1)
        users_final, items_final = torch.split(all_emb, [self.num_users, self.num_items], dim=0)
        return users_final, items_final

    # ---------- Popularity utils ----------
    def _compute_item_popularity(self) -> np.ndarray:
        if hasattr(self.dataset, "UserItemNet"):
            try:
                return np.asarray(self.dataset.UserItemNet.sum(axis=0)).ravel().astype(np.int64)
            except Exception:
                pass
        if hasattr(self.dataset, "trainUser"):
            counts = np.zeros(self.num_items, dtype=np.int64)
            for items in self.dataset.trainUser:
                for i in items: counts[int(i)] += 1
            return counts
        if hasattr(self.dataset, "allPos"):
            counts = np.zeros(self.num_items, dtype=np.int64)
            for u in range(self.num_users):
                for i in self.dataset.allPos(u): counts[int(i)] += 1
            return counts
        return np.ones(self.num_items, dtype=np.int64)

    @staticmethod
    def _bin_popularity(counts: np.ndarray, n_bins: int):
        logc = np.log1p(counts.astype(np.float64))
        edges = np.quantile(logc, np.linspace(0., 1., n_bins + 1))
        edges[0] -= 1e-8; edges[-1] += 1e-8
        bins = np.digitize(logc, edges[1:-1], right=False).astype(np.int64)
        return bins, edges

    def _fuse_item_with_pop(self, i_gcn: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        bins = self.item_pop_bin[item_ids]
        p    = self.pop_emb(bins)
        gate = torch.sigmoid(self.gate_i(i_gcn) + self.gate_p(p))
        return gate * i_gcn + (1.0 - gate) * p

    # ---------- Global context ----------
    def _global_context(self, users_emb, items_emb):
        return users_emb.mean(dim=0) + items_emb.mean(dim=0)

    # ---------- Scoring ----------
    def _score_with_mlp(self, u_vec, i_vec, users, items, users_emb, items_emb):
        if self.use_norm:
            u_vec = F.normalize(u_vec, dim=-1)
            i_vec = F.normalize(i_vec, dim=-1)
        u_b = self.bias_scale * self.user_bias(users.long())
        i_b = self.bias_scale * self.item_bias(items.long())
        g   = self._global_context(users_emb, items_emb).unsqueeze(0).expand(u_vec.size(0), -1)
        x   = torch.cat([u_vec, i_vec, u_b, i_b, g], dim=1)
        x   = self.pre_norm(x)
        mlp_score = self.mlp(x).squeeze(-1)
        if self.residual_alpha > 0.0:
            dot = torch.sum(u_vec * i_vec, dim=-1)
            return self.residual_alpha * dot + (1.0 - self.residual_alpha) * mlp_score
        return mlp_score

    # ---------- Forward ----------
    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        if self.training:
            users_emb, items_emb = self.computer()  # fresh graph
        else:
            if self._cached_users is None:
                self._cached_users, self._cached_items = self.computer()
            users_emb, items_emb = self._cached_users, self._cached_items

        u = users_emb[users.long()]
        i = items_emb[items.long()]
        if self.use_pop_gate:
            i = self._fuse_item_with_pop(i, items.long())
        return self._score_with_mlp(u, i, users, items, users_emb, items_emb)

    # ---------- Full scoring for evaluation ----------
    def getUsersRating(self, users: torch.Tensor) -> torch.Tensor:
        if self.training:
            users_emb, items_emb = self.computer()  # fresh
        else:
            if self._cached_users is None:
                self._cached_users, self._cached_items = self.computer()
            users_emb, items_emb = self._cached_users, self._cached_items

        u_gcn = users_emb[users.long()]  # [B, D]
        I = items_emb                    # [N, D]
        if self.use_pop_gate:
            all_item_ids = torch.arange(self.num_items, device=self.device, dtype=torch.long)
            I = self._fuse_item_with_pop(I, all_item_ids)
        if self.use_norm:
            u_gcn = F.normalize(u_gcn, dim=-1)
            I     = F.normalize(I, dim=-1)

        # Chunked MLP scoring
        B, N, D = u_gcn.size(0), I.size(0), I.size(1)
        chunk = 8192
        out = []
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            i_chunk = I[s:e]
            u_e = u_gcn.unsqueeze(1).expand(B, e - s, D)
            i_e = i_chunk.unsqueeze(0).expand(B, e - s, D)
            u_b = self.bias_scale * self.user_bias(users.long()).unsqueeze(1).expand_as(u_e)
            i_b = self.bias_scale * self.item_bias(torch.arange(s, e, device=self.device)).unsqueeze(0).expand_as(i_e)
            g   = self._global_context(users_emb, items_emb).unsqueeze(0).unsqueeze(0).expand(B, e - s, D)
            X   = torch.cat([u_e, i_e, u_b, i_b, g], dim=2)
            X   = self.pre_norm(X)
            scores = self.mlp(X).squeeze(-1)
            if self.residual_alpha > 0.0:
                dot = (u_e * i_e).sum(dim=-1)
                scores = self.residual_alpha * dot + (1.0 - self.residual_alpha) * scores
            out.append(scores)
        return torch.cat(out, dim=1)

    # ---------- BPR loss ----------
    def bpr_loss(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        users_emb, items_emb = self.computer()  # ALWAYS fresh each backward
        u  = users_emb[users.long()]
        ip = items_emb[pos.long()]
        ineg = items_emb[neg.long()]
        if self.use_pop_gate:
            ip   = self._fuse_item_with_pop(ip,  pos.long())
            ineg = self._fuse_item_with_pop(ineg, neg.long())
        if self.use_norm:
            u   = F.normalize(u, dim=-1)
            ip  = F.normalize(ip, dim=-1)
            ineg= F.normalize(ineg, dim=-1)

        up = self._score_with_mlp(u, ip, users, pos, users_emb, items_emb)
        un = self._score_with_mlp(u, ineg, users, neg, users_emb, items_emb)

        loss = torch.mean(F.softplus(un - up))
        reg  = (u.norm(2).pow(2) + ip.norm(2).pow(2) + ineg.norm(2).pow(2)) / (2.0 * len(users))
        return loss, reg

    # ---------- Cache control ----------
    def invalidate_cache(self):
        self._cached_users = None
        self._cached_items = None

    @torch.no_grad()
    def print_pop_gate_stats(self, sample_items=4096):
        if not self.use_pop_gate:
            print("[pop-gate] disabled"); return
        idx = torch.randint(0, self.num_items, (sample_items,), device=self.device)
        i = self.embedding_item.weight[idx]
        p = self.pop_emb(self.item_pop_bin[idx])
        gate = torch.sigmoid(self.gate_i(i) + self.gate_p(p))
        print(f"[pop-gate] mean={gate.mean().item():.3f}, std={gate.std().item():.3f}")
        for b in range(self.pop_bins):
            m = (self.item_pop_bin[idx] == b)
            if m.any():
                print(f"  bin {b}: mean={gate[m].mean().item():.3f}, n={int(m.sum())}")
