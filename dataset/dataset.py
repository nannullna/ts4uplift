from typing import List, Union, Optional, Dict, Any, Tuple
from collections import defaultdict
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence, # Packs a Tensor containing padded sequences of variable length
    pad_packed_sequence,  # Pads a packed batch of variable length sequences
)



def second_embedding(x: np.ndarray) -> np.ndarray:
    # unit is second in [0, 60)
    return np.stack([np.sin(2 * np.pi / 60 * x), np.cos(2 * np.pi / 60 * x)])

def daily_embedding(x: np.ndarray) -> np.ndarray:
    # unit is hour in [0, 24)
    return np.stack([np.sin(2 * np.pi / 24 * x), np.cos(2 * np.pi / 24 * x)])

def weekly_embedding(x: np.ndarray) -> np.ndarray:
    # unit is dayofweek in [0, 6]
    return np.stack([np.sin(2 * np.pi / 7 * x), np.cos(2 * np.pi / 7 * x)])


def default_time_transform(X: pd.Series) -> torch.Tensor:

    day_of_week = np.array(X.dt.dayofweek) # (L,)
    hour_of_day = np.array(X.dt.hour + X.dt.minute / 60)
    sec_of_hour = np.array(X.dt.second + X.dt.microsecond / 1e6)

    embedding_D = daily_embedding(hour_of_day) # (2, L)
    embedding_W = weekly_embedding(day_of_week) # (2, L)
    embedding_S = second_embedding(sec_of_hour) # (2, L)
    
    x = torch.tensor(np.concatenate([embedding_W, embedding_D, embedding_S], axis=0).T, dtype=torch.float32).contiguous() # (L, 6)

    return x


def binning_transform(df: pd.DataFrame, start, end, freq='10T'):
    """Binning transform for time series data"""
    bins = pd.date_range(start=start, end=end, freq=freq)
    time_cut = pd.cut(df['timestamp'], bins=bins, right=False)

    time_cut = time_cut.value_counts().sort_index()
    time_cut = time_cut.fillna(0)
    time_cut = torch.tensor(time_cut, dtype=torch.float32)

    time_emb = default_time_transform(bins[:-1].to_series())

    assert time_cut.shape[0] == time_emb.shape[0], f'{time_cut.shape[0]} != {time_emb.shape[0]}'

    return time_emb, time_cut


class UpliftDataset(Dataset):

    METHOD_TO_IDX = {'LOGIN': 0, 'POST': 1, 'GET': 2, 'PUT': 3, 'DELETE': 4}
    IDX_TO_METHOD = {0: 'LOGIN', 1: 'POST', 2: 'GET', 3: 'PUT', 4: 'DELETE'}

    def __init__(self, root: str, preprocess=None, time_transform=None, feature_transform=None, target_transform=None, y_idx: int=0) -> None:
        super().__init__()
        self.root = root
        try:
            self.info = pd.read_csv(os.path.join(self.root, 'info.csv'))
        except FileNotFoundError:
            self.info = pd.read_json(os.path.join(self.root, 'info.json'))
        
        self.preprocess = preprocess
        self.time_transform = time_transform if time_transform is not None else self.default_time_transform
        self.feature_transform = feature_transform if feature_transform is not None else self.default_feature_transform
        self.target_transform = target_transform  if target_transform is not None else self.default_target_transform

        self.target_y_idx = y_idx

    def __len__(self, select_indices: list=None) -> int:
        return len(self.info if select_indices is None else self.info.iloc[select_indices])


    def load_data(self, index: int):
        """Loads data from the dataset by index."""
        info = self.info.iloc[index]
        X = pd.read_parquet(os.path.join(self.root, info['X']))
        T = info['T']
        Y = info['Y']
        return X, T, Y


    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        
        X, T, Y = self.load_data(index)

        if self.preprocess is not None:
            timestamp, ftrs = self.preprocess(X, T, Y)

        else:
            timestamp = X['timestamp']
            timestamp = self.time_transform(timestamp)
            ftrs = self.feature_transform(X)
        
        if self.target_transform is not None:
            T, Y = self.target_transform(T, Y)
        else:
            T, Y = torch.tensor(T, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32) # (1,), (1,)
    
        return {'timestamp': timestamp, 'X': ftrs, 't': T, 'y': Y}


    def default_feature_transform(self, ftrs: pd.DataFrame) -> torch.Tensor:
        """Default transform for features."""
        x1 = torch.tensor(ftrs['action'], dtype=torch.long)
        x2 = torch.tensor(ftrs['method'].apply(lambda x: self.METHOD_TO_IDX[x]), dtype=torch.long)
        return torch.stack([x1, x2], dim=1) # (L, 2)


    def default_time_transform(self, X: pd.Series) -> torch.Tensor:
        return default_time_transform(X)

    
    def default_target_transform(self, T: int, Y: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Default transform for target."""
        return torch.tensor(T, dtype=torch.float), torch.tensor(Y[self.target_y_idx], dtype=torch.float) # (1,), (1,)


    def split(self, by: str='random', ratio: float=0.2, random_state: int=42, select_indices: list=None, **kwargs) -> Tuple[Subset, Subset]:
        """Splits the dataset into train and validation by user."""
        if by == 'random':
            train_ids, valid_ids = self.val_by_random(ratio, random_state, select_indices, **kwargs)
        elif by == 'user':
            train_ids, valid_ids = self.val_by_user(ratio, random_state, select_indices, **kwargs)
        elif by == 'match':
            train_ids, valid_ids = self.val_by_match(ratio, random_state, select_indices, **kwargs)
        elif by == 'test':
            train_ids, valid_ids = self.test_split(ratio, random_state, select_indices, **kwargs)
        else:
            raise ValueError(f'Invalid split method: {by}')
        return Subset(self, train_ids), Subset(self, valid_ids)


    def val_by_random(self, val_ratio: float, random_state: int=42, select_indices: list=None, **kwargs) -> Tuple[List[int], List[int]]:
        """Splits the dataset into train and validation in random fashion and returns indices. Do not consider overlap in users."""
        np.random.seed(random_state)
        unif = np.random.rand(self.__len__())
        ids = unif.argsort()
        train_size = int((1-val_ratio) * self.__len__())
        train_ids = sorted(ids[:int(train_size * self.__len__())].tolist())
        val_ids   = sorted(ids[int(train_size * self.__len__()):].tolist())
        raise (train_ids, val_ids)


    def split_users_by_treatment(self, df: pd.DataFrame) -> Dict[int, List[str]]:
        """Splits users by treatment into three groups.

        There are three types of users in the dataset.
            1) T=0 only exists. --> entirely added to T=0 group
            2) T=1 only exists. --> entirely added to T=1 group
            3) Both T=0 and T=1 exist.
        """
        # First, split users into three groups.
        gamer_to_group = defaultdict(int)
        for i, info in df.iterrows():
            gamer_id = info['X'].split('_')[0]
            t = info['T']
            gamer_to_group[gamer_id] += 2**t

        # Second, split users in each group into train and validation.
        group_to_gamers = {1: [], 2: [], 3: []}
        for k, v in gamer_to_group.items():
            group_to_gamers[v].append(k)
        
        return group_to_gamers


    def val_by_user(self, val_ratio: float, random_state: int=42, select_indices: list=None, **kwargs) -> Tuple[List[int], List[int]]:
        """Splits the dataset into train and validation by unique user in random fashion and returns indices.
        """
        group_to_gamers = self.split_users_by_treatment(self.info if select_indices is None else self.info.iloc[select_indices])
        
        np.random.seed(random_state)
        train_gamers, val_gamers = [], []
        for group_id, gamer_ids in group_to_gamers.items():
            print(f"Group {group_id}: {len(gamer_ids)} users.")

            gamer_ids_np = np.asarray(gamer_ids)

            total_size = len(gamer_ids)
            train_size = int((1-val_ratio) * total_size)
            unif = np.random.rand(total_size)
            ids = unif.argsort()
            train_gamers += gamer_ids_np[ids[:train_size]].tolist()
            val_gamers   += gamer_ids_np[ids[train_size:]].tolist()
        
        train_ids = [i for i, info in self.info.iterrows() if info['X'].split('_')[0] in train_gamers]
        val_ids   = [i for i, info in self.info.iterrows() if info['X'].split('_')[0] in val_gamers]

        return (train_ids, val_ids)

    
    def test_split(self, propensity_score: float=0.5, random_state: int=42, select_indices: list=None, **kwargs) -> Tuple[List[int], List[int]]:
        """Splits users in both T=1 and T=0 groups in random fashion for no overlap in users. No validation set."""
        group_to_gamers = self.split_users_by_treatment(self.info if select_indices is None else self.info.iloc[select_indices])
        
        # First handle users in both groups.
        gamers_in_both_np = np.asarray(group_to_gamers[3])

        np.random.seed(random_state)
        unif = np.random.rand(len(gamers_in_both_np))
        ids = unif.argsort()
        t1_size = int(propensity_score * len(gamers_in_both_np))
        gamers_in_both_t1 = gamers_in_both_np[ids[:t1_size]].tolist()
        gamers_in_both_t0 = gamers_in_both_np[ids[t1_size:]].tolist()

        test_ids = [i for i, info in self.info.iterrows() if (info['X'].split('_')[0] in gamers_in_both_t1) and (info['T'] == 1)]
        test_ids += [i for i, info in self.info.iterrows() if (info['X'].split('_')[0] in gamers_in_both_t0) and (info['T'] == 0)]

        # Second handle users in either group.
        gamers_in_t0 = group_to_gamers[1]
        gamers_in_t1 = group_to_gamers[2]

        test_ids += [i for i, info in self.info.iterrows() if info['X'].split('_')[0] in gamers_in_t1]
        test_ids += [i for i, info in self.info.iterrows() if info['X'].split('_')[0] in gamers_in_t0]

        return (test_ids, [])
    

    def val_by_match(self, val_ratio: float, random_state: int=42, select_indices: list=None, **kwargs) -> Tuple[List[int], List[int]]:
        """Splits the dataset into train and validation by user in match fashion and returns indices."""
        raise NotImplementedError


def collate_fn(data, max_length: int=1024, pad_on_right: bool=True):
    time = [d['timestamp'] for d in data]
    X = [d['X'] for d in data]
    t = [d['t'] for d in data]
    y = [d['y'] for d in data]

    # Pad to the left
    max_len = max([len(x) for x in X]) # max length of time series
    X = [
        torch.cat([torch.zeros(max_len - len(x), x.shape[1], dtype=torch.long, device=x.device), x]) 
        if not pad_on_right else
        torch.cat([x, torch.zeros(max_len - len(x), x.shape[1], dtype=torch.long, device=x.device)])
        for x in X
    ]
    X = torch.stack(X, dim=0) # (B, L, D_x)

    time = [
        torch.cat([torch.zeros(max_len - len(tm), tm.shape[1], dtype=torch.long, device=tm.device), tm])
        if not pad_on_right else
        torch.cat([tm, torch.zeros(max_len - len(tm), tm.shape[1], dtype=torch.long, device=tm.device)])
        for tm in time
    ]
    time = torch.stack(time, dim=0) # (B, L, D_time)

    if time.size(1) > max_length:
        time = time[:, -max_length:, :]
        X = X[:, -max_length:, :]

    t = torch.stack(t, dim=0) # (B,)
    y = torch.stack(y, dim=0) # (B,)
    
    return {
        'timestamp': time,
        'X': X,
        't': t,
        'y': y
    }