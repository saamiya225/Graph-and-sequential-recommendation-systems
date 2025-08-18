# code/dataloader.py
"""
Dataset and graph utilities for LightGCN—compatible with this repo.
Reads train.txt/test.txt under ../data/<dataset>/ and builds the bipartite graph.
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
    Compatibility shell. In this fork we use Loader as the concrete dataset.
    """
    def __init__(self):
        pass

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getSparseGraph(self):
        raise NotImplementedError


class Loader(BasicDataset):
    """
    Concrete dataset loader: reads <path>/train.txt and <path>/test.txt
    and builds normalized adjacency for LightGCN.
    """
    def __init__(self, config=world.config, path=None):
        # Resolve dataset path
        if path is None:
            path = join(world.DATA_PATH, world.dataset)
        self.path = path

        cprint(f'loading [{self.path}]')
        self.split = config.get('A_split', False)
        self.folds = config.get('A_n_fold', 100)

        self.mode_dict = {'train': 0, 'test': 1}
        self.mode = self.mode_dict['train']

        # will be determined while parsing
        self.n_user = 0
        self.m_item = 0

        train_file = join(self.path, 'train.txt')
        test_file  = join(self.path, 'test.txt')

        trainUniqueUsers, trainUsers, trainItems = [], [], []
        testUniqueUsers,  testUsers,  testItems  = [], [], []

        self.traindataSize = 0
        self.testDataSize  = 0

        # ----- read train -----
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

        # ----- read test -----
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

        # +1 because indices are zero-based
        self.n_user += 1
        self.m_item += 1

        self.trainUniqueUsers = np.array(trainUniqueUsers, dtype=np.int64)
        self.trainUser = np.array(trainUsers, dtype=np.int64)
        self.trainItem = np.array(trainItems, dtype=np.int64)

        self.testUniqueUsers = np.array(testUniqueUsers, dtype=np.int64)
        self.testUser = np.array(testUsers, dtype=np.int64)
        self.testItem = np.array(testItems, dtype=np.int64)

        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items:.12f}")

        # Build (users,items) bipartite matrix
        self.UserItemNet = sp.csr_matrix(
            (np.ones(len(self.trainUser), dtype=np.float32), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item)
        )
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1.
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        # Precompute all positives and test dict
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

        self.Graph = None  # will cache adj (or folds) after getSparseGraph()

    # ---------- properties ----------
    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    # ---------- helpers ----------
    def __build_test(self):
        """
        Returns dict: {user: [items]}
        """
        test_data = {}
        for u, i in zip(self.testUser, self.testItem):
            if u in test_data:
                test_data[u].append(i)
            else:
                test_data[u] = [i]
        return test_data

    def getUserItemFeedback(self, users, items):
        # returns a vector of 0/1 for whether (u,i) exists in train
        arr = np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))
        return arr

    def getUserPosItems(self, users):
        posItems = []
        for u in users:
            posItems.append(self.UserItemNet[u].indices)
        return posItems

    # ---------- adjacency ----------
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.from_numpy(coo.row).long()
        col = torch.from_numpy(coo.col).long()
        index = torch.stack([row, col], dim=0)
        data = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

    def _split_A_hat(self, A):
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
        Build (or load) the normalized adjacency for LightGCN:
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
            adj_mat = sp.dok_matrix((n_all, n_all), dtype=np.float32)
            adj_mat = adj_mat.tolil()

            R = self.UserItemNet.tolil()
            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.tocsr()

            # symmetric normalize: D^{-1/2} A D^{-1/2}
            rowsum = np.array(adj_mat.sum(axis=1)).flatten()
            d_inv = np.power(rowsum, -0.5, where=rowsum!=0)
            d_inv[np.isinf(d_inv)] = 0.
            D_inv = sp.diags(d_inv)

            norm_adj = D_inv.dot(adj_mat).dot(D_inv).tocsr()

            print(f"costing {time() - s:.2f}s, saved norm_mat...")
            sp.save_npz(pre_adj_path, norm_adj)

        if self.split:
            self.Graph = self._split_A_hat(norm_adj)
            print("done split matrix")
        else:
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj).coalesce().to(world.device)
            print("don't split the matrix")
        return self.Graph

    # ---------- dataset protocol ----------
    def __getitem__(self, idx):
        # used by sampler (e.g., returns a user index)
        return self.trainUniqueUsers[idx]

    def __len__(self):
        # number of users with at least one interaction in train
        return len(self.trainUniqueUsers)
