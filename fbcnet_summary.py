"""
File: fbcnet_summary.py
Author: Chuncheng Zhang
Date: 2025-09-26
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Summary the fbcnet results.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-09-26 ------------------------
# Requirements and constants
from sklearn import metrics

from util.easy_import import *
from util.io.file import load


# %%
DATA_DIR = Path('./results/fbcnet-gpu')

# %%
dump_files = sorted(list(DATA_DIR.rglob('*.dump')))

# %% ---- 2025-09-26 ------------------------
# Function and class


# %% ---- 2025-09-26 ------------------------
# Play ground
objs = [load(p) for p in dump_files]

y_true = []
y_pred = []
reports = []
for obj in objs:
    print(obj)
    y_true.append(obj['y_true'])
    y_pred.append(obj['y_pred'])
    report = metrics.classification_report(
        y_true=obj['y_true'], y_pred=obj['y_pred'], output_dict=True)
    reports.append(report)

y_true = np.array(y_true).ravel()
y_pred = np.array(y_pred).ravel()

for report in reports:
    print(report['accuracy'])
print(metrics.classification_report(y_true=y_true, y_pred=y_pred))


# %% ---- 2025-09-26 ------------------------
# Pending


# %% ---- 2025-09-26 ------------------------
# Pending
