"""
File: fbcnet.py
Author: Chuncheng Zhang
Date: 2025-09-26
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Decoding with fbcnet.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-09-26 ------------------------
# Requirements and constants
import mne

import torch
import torch.nn as nn
import torch.optim as optim
from torcheeg.models import FBCNet

from util.easy_import import *
from util.io.file import save
from read_data_link import MyData

# %%
DATA_DIR = Path('./raw/20251029')
OUTPUT_DIR = Path('./results/fbcnet-gpu')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %%
cnt_files = sorted(list(DATA_DIR.rglob('*.cnt')))
edf_files = sorted(list(DATA_DIR.rglob('*.edf')))
file_pairs = [{'cnt': c, 'edf': e} for (c, e) in zip(cnt_files, edf_files)]

# Load data
mds = []
for pair in tqdm(file_pairs, 'Load data'):
    md = MyData(pair['cnt'])
    md.link_to_edf(pair['edf'])
    mds.append(md)

groups = np.concatenate([
    np.zeros((len(md.epochs), )) + i
    for i, md in enumerate(mds)])


# %%
FREQ_BANDS = [(8, 12), (12, 16), (16, 20),
              (20, 24), (24, 28), (28, 32)]

# %% ---- 2025-09-26 ------------------------
# Function and class


class DataLoader:
    def __init__(self, X, y, groups, test_group=0):
        self.X = X
        # Scale into 1 scale
        self.X /= np.max(np.abs(self.X))
        self.y = y

        self.X = torch.tensor(self.X).cuda()
        self.y = torch.tensor(self.y).cuda()

        self.groups = groups
        # Separate groups
        unique_groups = sorted(np.unique(self.groups).tolist())
        self.test_groups = [test_group]
        self.train_groups = [
            e for e in unique_groups if not e in self.test_groups]
        logger.info(
            f'DataLoader: {self.X.shape = }, {self.y.shape = }, {self.groups.shape = }, {self.train_groups = }, {self.test_groups = }')

    def yield_train_data(self, batch_size=32):
        train_idx = [g in self.train_groups for g in self.groups]
        while True:
            X = self.X[train_idx]
            y = self.y[train_idx]
            n_samples = X.shape[0]
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                yield X[batch_indices], y[batch_indices]

    def get_test_data(self):
        test_idx = [g in self.test_groups for g in self.groups]
        X = self.X[test_idx]
        y = self.y[test_idx]
        return X, y


# %% ---- 2025-09-26 ------------------------
# Play ground
# Concat epochs
epochs = mne.concatenate_epochs([md.epochs for md in mds])
event_ids = list(epochs.event_id.keys())
epochs.load_data()
print(epochs)

# %%

# Generate X, y
X = epochs.get_data(copy=False)
y = epochs.events[:, -1]


# %%
# Filter and stack X
new_X = []
for low_freq, high_freq in tqdm(FREQ_BANDS, 'Filtering'):
    # 频带滤波
    X_filtered = X.copy().astype(np.float64, copy=False)
    for i in range(X.shape[0]):
        # 创建Epochs对象进行滤波
        _epochs = mne.EpochsArray(X[i:i+1], epochs.info)
        epochs_filtered = _epochs.filter(l_freq=low_freq, h_freq=high_freq,
                                         method='iir', verbose=False)
        X_filtered[i] = epochs_filtered.get_data()[0]
    new_X.append(X_filtered)

# new_X shape (n_bands, n_samples, n_channels, n_times)
new_X = np.array(new_X)

# Convert into (n_samples, n_bands, n_electrodes, n_times)
X = new_X.transpose((1, 0, 2, 3))

print(f'{X.shape=}')

# %% ---- 2025-09-26 ------------------------
for test_group in np.unique(groups):
    # Make model
    # shape is (n_samples, n_bands, n_electrodes, n_times)
    shape = X.shape
    num_classes = len(np.unique(y))

    # Model
    model = FBCNet(
        num_electrodes=shape[2],
        chunk_size=600,
        in_channels=shape[1],
        num_classes=num_classes,
    ).cuda()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 多分类任务常用
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(model, criterion, optimizer)

    # Training loop
    dl = DataLoader(X[:, :, :, :600], y, groups, test_group=test_group)
    it = iter(dl.yield_train_data(batch_size=64))

    for epoch in tqdm(range(5000), desc='Epoch'):
        def _train():
            X, y = next(it)
            # print(f'{X.shape=}, {y.shape=}')

            _y = model(torch.tensor(X, dtype=torch.float32))
            # print(f'{_y.shape=}')
            # print(_y)

            # 前向传播
            # loss = criterion(_y, torch.tensor(y-1))
            loss = criterion(_y, y-1)
            # print(f'{loss.item()=}')

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Report
            if epoch % 100 == 0:
                logger.info(f'Epoch {epoch}, Loss: {loss.item():.6f}')

        _train()

    # Testing loop
    def _test():
        X, y = dl.get_test_data()
        y_true = y.cpu().numpy()
        with torch.no_grad():
            _y = model(torch.tensor(X, dtype=torch.float32)).cpu()
            y_pred = torch.argmax(_y, dim=1).numpy() + 1
            accuracy = np.mean(y_pred == y_true)
            logger.info(f'Test Accuracy ({test_group}): {accuracy * 100:.2f}%')

        result = {
            'y_true': y_true,
            'y_pred': y_pred,
            'test_group': test_group
        }
        save(result, OUTPUT_DIR.joinpath(f'result-{test_group}.dump'))

    _test()

exit(0)
# %% ---- 2025-09-26 ------------------------
# Pending

# %%
