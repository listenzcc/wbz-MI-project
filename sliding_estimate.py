"""
File: sliding_estimate.py
Author: Chuncheng Zhang
Date: 2025-09-25
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Sliding estimate on the data.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-09-25 ------------------------
# Requirements and constants
import pandas as pd
import seaborn as sns
import mne
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

from mne.decoding import (
    CSP,
    Scaler,
    Vectorizer,
    LinearModel,
    SlidingEstimator,
    GeneralizingEstimator,
    get_coef,
    cross_val_multiscore,
)

from util.easy_import import *
from read_data import MyData

# %%
raw_directory = Path('./raw')
cnt_files = sorted(list(raw_directory.rglob('*.cnt')))
print(f'{cnt_files=}')

output_directory = Path('./results/decoding')
output_directory.mkdir(exist_ok=True, parents=True)

# %% ---- 2025-09-25 ------------------------
# Function and class


# %% ---- 2025-09-25 ------------------------
# Load data
mds = []
for p in tqdm(cnt_files, 'Load data'):
    md = MyData(p)
    mds.append(md)

groups = np.concatenate([
    np.zeros((len(md.epochs), )) + i
    for i, md in enumerate(mds)])

# Concat epochs
epochs = mne.concatenate_epochs([md.epochs for md in mds])
event_ids = list(epochs.event_id.keys())
epochs.load_data()
epochs

# Generate X, y
X = epochs.get_data()
y = epochs.events[:, -1]

# %%
# Over time decoding
clf = make_pipeline(
    StandardScaler(),
    LinearModel(LogisticRegression(solver="liblinear"))
)

scoring = make_scorer(accuracy_score, greater_is_better=True)

time_decod = SlidingEstimator(
    clf, n_jobs=n_jobs, scoring=scoring, verbose=True)

cv = LeaveOneGroupOut()

scores = cross_val_multiscore(
    time_decod, X, y, groups=groups, cv=cv, n_jobs=n_jobs)

# Mean scores across cross-validation splits
avg_scores = np.mean(scores, axis=0)
print(f'{avg_scores=}')

# %%
# 创建DataFrame
df = pd.DataFrame(scores.T)  # 转置以便每列是一个试验
df['Time'] = epochs.times

# 转换为长格式
df_long = df.melt(id_vars=['Time'], var_name='Trial', value_name='Score')

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_long, x='Time', y='Score', errorbar='sd', linewidth=2)
plt.axhline(y=0.5, color='red', linestyle='--',
            linewidth=1, alpha=0.7, label='y=0.5')
plt.axvline(x=0.0, color='red', linestyle='--',
            linewidth=1, alpha=0.7, label='x=0.0')
plt.title('Scores over Time with Standard Deviation')
plt.savefig(output_directory.joinpath('scores-over-time.png'))
plt.show()

# %% ---- 2025-09-25 ------------------------
# Pending

# %% ---- 2025-09-25 ------------------------
# Pending

# %%
