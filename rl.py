import os
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import TensorDataset, DataLoader, Sampler

def make_df(df, target_reward, train_id, valid_id, test_id):
    train_df = pd.merge(df, train_id['traj'], on='traj', how='inner')
    val_df = pd.merge(df, valid_id['traj'], on='traj', how='inner')
    test_df = pd.merge(df, test_id['traj'], on='traj', how='inner')

    train_df.loc[(train_df[target_reward] != -1) & (train_df[target_reward] != 1), target_reward] = 0
    val_df.loc[(val_df[target_reward] != -1) & (val_df[target_reward] != 1), target_reward] = 0
    test_df.loc[(test_df[target_reward] != -1) & (test_df[target_reward] != 1), target_reward] = 0

    val_df['Dead'] = val_df[target_reward]
    test_df['Dead'] = test_df[target_reward]

    return train_df, val_df, test_df

def make_transition(df, target, rolling_size):
    s_col = [x for x in df if x[:2] == 's:']
    a_col = [x for x in df if x[:2] == 'a:']

    dict = {}
    dict['traj'] = {}

    s, a, r, s2, t = [], [], [], [], []

    for traj in tqdm(df.traj.unique()):
        df_traj = df[df['traj'] == traj]
        dict['traj'][traj] = {'s': [], 'a': [], 'r': []}
        dict['traj'][traj]['s'] = df_traj[s_col].values.tolist()
        dict['traj'][traj]['a'] = df_traj[a_col].values.tolist()
        dict['traj'][traj]['r'] = df_traj[target].values.tolist()

        step_len = len(df_traj) - rolling_size

        for step in range(step_len):
            s.append(dict['traj'][traj]['s'][step + rolling_size - 1])
            a.append(dict['traj'][traj]['a'][step + rolling_size - 1])
            r.append(dict['traj'][traj]['r'][step + rolling_size])
            s2.append(dict['traj'][traj]['s'][step + rolling_size])
            if (dict['traj'][traj]['r'][step + rolling_size] in [-1, 1]) or (step == (step_len - 1)):
                t.append(1)
                break
            else:
                t.append(0)

    s = torch.FloatTensor(np.float32(s))
    a = torch.LongTensor(np.int64(a))
    r = torch.FloatTensor(np.int64(r))
    s2 = torch.FloatTensor(np.float32(s2))
    t = torch.FloatTensor(np.float32(t))
    Dataset = TensorDataset(s, a, r, s2, t)

    return Dataset

def make_transition_test(df, target, rolling_size=24, batch_size=4096):
    s_col = [x for x in df if x[:2] == 's:']
    a_col = [x for x in df if x[:2] == 'a:']

    dict = {}
    dict['traj'] = {}

    s, a, dead, patients = [], [], [], []

    for traj in tqdm(df.traj.unique()):
        df_traj = df[df['traj'] == traj]
        dict['traj'][traj] = {'s': [], 'a': [], 'pa': [], 'r': []}
        dict['traj'][traj]['s'] = df_traj[s_col].values.tolist()
        dict['traj'][traj]['a'] = df_traj[a_col].values.tolist()
        dict['traj'][traj]['r'] = df_traj[target].values.tolist()

        final_status = sum(dict['traj'][traj]['r'])
        step_len = len(df_traj) - rolling_size

        for step in range(step_len):
            s.append(dict['traj'][traj]['s'][step + rolling_size - 1])
            a.append(dict['traj'][traj]['a'][step + rolling_size - 1])
            dead.append(dict['traj'][traj]['r'][step + rolling_size])
            patients.append(final_status)

    s = torch.FloatTensor(np.float32(s)).squeeze(1)
    a = torch.LongTensor(np.int64(a)).squeeze(1)
    dead = torch.FloatTensor(np.int64(dead))
    patients = torch.LongTensor(np.int64(patients))

    Dataset = TensorDataset(s, a, dead, patients)
    rt = DataLoader(Dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return rt

class Sampler(Sampler):
    def __init__(self, data, batch_size, version, ns):
        self.data = data
        self.batch_size = batch_size
        self.num_samples_1 = ns
        self.num_samples_0 = batch_size - ns

        target_values = []
        baseline_values = []

        if version == '_negative':
            target_values.append(-1.0)
            baseline_values.append(+1.0)
            baseline_values.append(0.0)
        if version == '_positive':
            target_values.append(+1.0)
            baseline_values.append(-1.0)
            baseline_values.append(0.0)
        if version == '_both':
            target_values.append(+1.0)
            target_values.append(-1.0)
            baseline_values.append(0.0)

        self.indices = [i for i in range(len(data)) if data[i][2].item() in baseline_values]
        self.indices_neg = [i for i in range(len(data)) if data[i][2].item() in target_values]

        self.used_indices_neg = []

    def __iter__(self):
        np.random.shuffle(self.indices)
        np.random.shuffle(self.indices_neg)

        batch = []
        for idx in self.indices:
            batch.append(idx)
            if len(batch) == self.num_samples_0:
                batch.extend(self._sample_indices_neg())
                yield batch
                batch = []

    def _sample_indices_neg(self, remaining=0):
        if remaining:
            num_samples_1 = self.batch_size - remaining
        else:
            num_samples_1 = self.num_samples_1

        if len(self.used_indices_neg) + num_samples_1 > len(self.indices_neg):
            self.used_indices_neg = []
            np.random.shuffle(self.indices_neg)

        indices_neg = self.indices_neg[len(self.used_indices_neg):len(self.used_indices_neg) + num_samples_1]
        self.used_indices_neg.extend(indices_neg)
        return indices_neg

    def __len__(self):
        return (len(self.indices) + len(self.indices_neg)) // self.batch_size

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.)

class Model(nn.Module):
    def __init__(self, obs_dim, mlp_size, mlp_num_layers, nb_actions, activation_type):
        super(Model, self).__init__()
        activation_functions = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'Tanh': nn.Tanh(),
            'ELU': nn.ELU(),
            'CELU': nn.CELU()
        }
        activation_function = activation_functions.get(activation_type, nn.ReLU())
        self.mlp = nn.ModuleList()
        for i in range(mlp_num_layers):
            if i == 0:
                self.mlp.append(nn.Sequential(
                    nn.Linear(obs_dim, mlp_size),
                    activation_function
                ))
            else:
                self.mlp.append(nn.Sequential(
                    nn.Linear(mlp_size, mlp_size),
                    activation_function
                ))
            if i == mlp_num_layers - 1:
                self.mlp.append(nn.Sequential(
                    nn.Linear(mlp_size, nb_actions)
                ))
        self.mlp.apply(init_weights)

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x