"""
File: read_data.py
Author: Chuncheng Zhang
Date: 2025-09-25
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read raw data.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-09-25 ------------------------
# Requirements and constants
import mne
from util.easy_import import *

# %% ---- 2025-09-25 ------------------------
# Function and class


class StandardMontage:
    montage = mne.channels.make_standard_montage('standard_1020')
    rename_mapping = {e: e.upper() for e in montage.ch_names}

    def __init__(self):
        self.montage.rename_channels(self.rename_mapping)


class MySetup:
    epochs = {
        'tmin': -1,
        'tmax': 4,
        'decim': 5
    }


class MyData:
    montage = StandardMontage().montage
    setup = MySetup()

    def __init__(self, fpath: Path):
        self.fpath = fpath
        self.raw = self.read_raw()
        self.epochs, self.events, self.event_id = self.mk_epochs()
        logger.info(f'Read from {self.fpath=}, {self.raw=}, {self.epochs=}')

    def read_raw(self):
        raw = mne.io.read_raw_cnt(self.fpath)
        raw.pick([e for e in raw.ch_names if e in self.montage.ch_names])
        raw.set_montage(self.montage, on_missing='raise')
        return raw

    def mk_epochs(self):
        events, event_id = mne.events_from_annotations(self.raw)
        epochs = mne.Epochs(self.raw, events, event_id, **self.setup.epochs)
        epochs.drop_bad()
        return epochs, events, event_id


# %% ---- 2025-09-25 ------------------------
# Play ground


# %% ---- 2025-09-25 ------------------------
# Pending

# %% ---- 2025-09-25 ------------------------
# Pending


# %%
