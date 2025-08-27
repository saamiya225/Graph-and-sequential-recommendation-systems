# code/dataloader.py
"""
Dataset and graph utilities for LightGCN — active and required in this repo.

This script is responsible for:
1. Reading `train.txt` and `test.txt` from the dataset folder.
2. Constructing the bipartite user–item graph.
3. Providing methods for accessing positive interactions, test sets, 
   and normalized adjacency matrices for LightGCN training.

Unlike the older "PPR weight" or "cache graph" scripts, THIS FILE IS NEEDED.
It directly feeds the model with graph structure and user–item interactions.
"""

import os
from os.path import join
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
import world
from world import cprint
from time import time


class BasicDataset(Dataset):
    """
    Abstract dataset wrapper for LightGCN.
    - Provides properties (n_users, m_items, etc.) that subclasses must implement.
    - Loader (below) implements all of these for real data.
    """
    def __init__(self):
        pass

    @property
    def n_users(self): raise NotImplementedError
    @property
    def m_items(self): raise NotImplementedError
    @property
    def trainDataSize(self): raise NotImplementedError
    @property
    def testDict(self): raise NotImplementedError
    @property
    def allPos(self): raise NotImplementedError

    def getUserItemFeedback(self, users, items): raise NotImplementedError
    def getUserPosItems(self, users): raise NotImplementedError
    def getSparseGraph(self): raise NotImplementedError


class Loader(BasicDataset):
    """
    Main dataset loader used in our pipeline.

    Responsibilities:
    - Reads interactions from `<dataset>/train.txt` and `<dataset>/test.txt`.
    - Builds user–item bipartite interaction matrix.
    - Provides normalized adjacency for LightGCN propagation.
    - Stores per-user positives (train) and test dictionaries.
    """

    def __init__(self, config=world.config, path=None):
        # Resolve dataset path
        if path is None:
            path = join(world.DATA_PATH, world.dataset)
        self.path = path
        cprint(f'loading [{self.path}]')

        # Config flags (from world.config)
        self.split = config.get('A_split', False)   # whether to split adjacency
        self.folds = config.get('A_n_fold', 100)    # number of folds if split

        # --- storage placeholders ---
        self.n_user, self.m_item = 0, 0
        self.traindataSize, self.testDataSize = 0, 0

        # Containers for training/test interactions
        trainUniqueUsers, trainUsers, trainItems = [], [], []
        testUniqueUsers,  testUsers,  testItems  = [], [], []

        # --- parse train file ---
        train_file = join(self.path, 'train.txt')
        with open(train_file, 'r') as f:
            for l in f:
                if not l.strip():
                    continue
                cols = l.strip().split()
                uid = int(cols[0])
                items = [int(i) for i in cols[1:]]
                if not items:
                    continue
                self.n_user = max(self.n_user, uid)
                self.m_item = max(self.m_item, max(items))
                trainUniqueUsers.append(uid)
                trainUsers.extend([uid] * len(items))
                trainItems.extend(items)
                self.traindataSize += len(items)

        # --- parse test file ---
        test_file = join(self.path, 'test.txt')
        with open(test_file, 'r') as f:
            for l in f:
                if not l.strip():
                    continue
                cols = l.strip().split()
                uid = int(cols[0])
                items = [int(i) for i in cols[1:]]
                if not items:
                    continue
                self.n_user = max(self.n_user, uid)
                self.m_item = max(self.m_item, max(items))
                testUniqueUsers.append(uid)
                testUsers.extend([uid] * len(items))
                testItems.extend(items)
                self.testDataSize += len(items)

        # Convert to numpy arrays
        self.n_user += 1
        self.m_item += 1
        self.trainUniqueUsers = np.array(trainUniqueUsers, dtype=np.int64)
        self.trainUser = np.array(trainUsers, dtype=np.int64)
        self.trainItem = np.array(trainItems, dtype=np.int64)
        self.testUniqueUsers = np.array(testUniqueUsers, dtype=np.int64)
        self.testUser = np.array(testUsers, dtype=np.int64)
        self.testItem = np.array(testItems, dtype=np.int64)

        # Logging
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items:.12f}")

        # Build CSR user–item interaction matrix
        self.UserItemNet = sp.csr_matrix(
            (np.ones(len(self.trainUser), dtype=np.float32), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item)
        )

        # Degrees for normalization
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1.
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        # Precompute positives and test dict
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

        # Graph cache placeholder
        self.Graph = None

    # ---------- basic properties ----------
    @property
    def n_users(self): return self.n_user
    @property
    def m_items(self): return self.m_item
    @property
    def trainDataSize(self): return self.traindataSize
    @property
    def testDict(self): return self.__testDict
    @property
    def allPos(self): return self._allPos

    # ---------- helpers ----------
    def __build_test(self):
        """Build {user: [test_items]} dict for evaluation."""
        test_data = {}
        for u, i in zip(self.testUser, self.testItem):
            if u in test_data: test_data[u].append(i)
            else: test_data[u] = [i]
        return test_data

    def getUserItemFeedback(self, users, items):
        """Binary feedback: 1 if (u,i) in train interactions, else 0."""
        arr = np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))
        return arr

    def getUserPosItems(self, users):
        """Return positive items for each user from training set."""
        return [self.UserItemNet[u].indices for u in users]

    # ---------- adjacency ----------
    def _convert_sp_mat_to_sp_tensor(self, X):
        """Convert scipy sparse to torch sparse tensor."""
        coo = X.tocoo().astype(np.float32)
        row = torch.from_numpy(coo.row).long()
        col = torch.from_numpy(coo.col).long()
        index = torch.stack([row, col], dim=0)
        data = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

    def _split_A_hat(self, A):
        """Optionally split adjacency into folds for GPU memory efficiency."""
        A_fold = []
        n_all = self.n_users + self.m_items
        fold_len = n_all // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            end = n_all if i_fold == self.folds - 1 else (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def getSparseGraph(self):
        """
        Build or load normalized adjacency for LightGCN:
            A = [0, R; R^T, 0]
            L = D^{-1/2} A D^{-1/2}
        """
        print("loading adjacency matrix")
        if self.Graph is not None:
            return self.Graph

        pre_adj_path = join(self.path, 's_pre_adj_mat.npz')
        try:
            norm_adj = sp.load_npz(pre_adj_path)
            print("successfully loaded...")
        except Exception:
            print("generating adjacency matrix")
            s = time()
            n_all = self.n_users + self.m_items

            # Build user–item bipartite adjacency
            adj_mat = sp.dok_matrix((n_all, n_all), dtype=np.float32).tolil()
            R = self.UserItemNet.tolil()
            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.tocsr()

            # Symmetric normalization
            rowsum = np.array(adj_mat.sum(axis=1)).flatten()
            d_inv = np.power(rowsum, -0.5, where=rowsum!=0)
            d_inv[np.isinf(d_inv)] = 0.
            D_inv = sp.diags(d_inv)
            norm_adj = D_inv.dot(adj_mat).dot(D_inv).tocsr()

            print(f"costing {time() - s:.2f}s, saved norm_mat...")
            sp.save_npz(pre_adj_path, norm_adj)

        # Cache in memory (split or not)
        if self.split:
            self.Graph = self._split_A_hat(norm_adj)
            print("done split matrix")
        else:
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj).coalesce().to(world.device)
            print("don't split the matrix")
        return self.Graph

    # ---------- dataset protocol ----------
    def __getitem__(self, idx):
        """Return user index (for sampling)."""
        return self.trainUniqueUsers[idx]

    def __len__(self):
        """Number of users with at least one training interaction."""
        return len(self.trainUniqueUsers)
