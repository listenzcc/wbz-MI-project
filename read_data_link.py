"""
File: read_data_link.py
Author: Chuncheng Zhang
Date: 2025-09-29
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read linking data for cnt and edf.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-09-29 ------------------------
# Requirements and constants
import mne
import pyedflib
from util.easy_import import *

# %%
DATA_DIR = Path('./raw/20250929')

# %% ---- 2025-09-29 ------------------------
# Function and class


class MySetup:
    epochs = {
        'tmin': -1,
        'tmax': 4,
        'decim': 8  # 5
    }


class StandardMontage:
    montage = mne.channels.make_standard_montage('standard_1020')
    rename_mapping = {e: e.upper() for e in montage.ch_names}

    def __init__(self):
        self.montage.rename_channels(self.rename_mapping)


def read_edf_file(fpath: Path):
    file = pyedflib.EdfReader(fpath.as_posix())
    fs = file.getSampleFrequencies()[0]
    channel_labels = file.getSignalLabels()
    signals = []
    for i in range(file.signals_in_file):
        signal = file.readSignal(i)
        signals.append(signal)

    dt = 1/fs

    return {
        'sfreq': fs,
        'ch_names': channel_labels,
        'data': np.array(signals),
        'times': np.array([e*dt for e in range(len(signals[0]))])
    }


def find_linked_files():
    cnt_files = sorted(list(DATA_DIR.rglob('*.cnt')))
    edf_files = sorted(list(DATA_DIR.rglob('*.edf')))
    file_pairs = [{'cnt': c, 'edf': e} for (c, e) in zip(cnt_files, edf_files)]
    return file_pairs


class MyData:
    montage = StandardMontage().montage
    setup = MySetup()

    def __init__(self, cnt_path: Path):
        self.cnt_path = cnt_path
        self.read_raw()

    def link_to_edf(self, edf_path: Path):
        self.edf_path = edf_path
        edf = read_edf_file(edf_path)
        markers = edf['data'][-1]
        times = edf['times']
        events = []
        for i, (m, t) in enumerate(zip(markers,  times)):
            if m == 0:
                continue
            marker = int(m/1000)
            delay = m % 1000
            real_t = t-delay/1000
            idx = len(times[times < real_t])
            events.append([idx, 0, marker])
        events = np.array(events)
        events[:, -1] = self.events[:, -1]
        ratios = np.diff(events[:, 0]) / np.diff(self.events[:, 0])
        self.marker_quality = np.mean(
            ratios - edf['sfreq'] / self.raw.info['sfreq'])

        # Link two objects
        raw = self.raw.copy()
        raw.pick([e.upper() for e in edf['ch_names'][:-1]])
        raw.resample(edf['sfreq'])
        self.edf_raw = mne.io.RawArray(edf['data'][:-1], raw.info)
        logger.info(f'Linked to {edf_path=}, {self.marker_quality=}')

        # Generate epochs
        self.epochs = mne.Epochs(
            self.raw, events, self.event_id, **self.setup.epochs)
        self.epochs.drop_bad()
        self.epochs.load_data()
        self.epochs.filter(l_freq=0, h_freq=40)

        return self.epochs

    def read_raw(self):
        raw = mne.io.read_raw_cnt(self.cnt_path)

        events, event_id = mne.events_from_annotations(raw)

        # Only use the ch_names inside the montage
        raw.pick([e for e in raw.ch_names if e in self.montage.ch_names])
        raw.set_montage(self.montage, on_missing='raise')

        self.events = events
        self.event_id = event_id
        self.raw = raw

        logger.info(f'Read from {self.cnt_path=}, {self.raw=}')
        return self.raw


# %% ---- 2025-09-29 ------------------------
# Play ground
if __name__ == '__main__':
    file_pairs = find_linked_files()
    print(file_pairs)

    for pair in tqdm(file_pairs, 'Read data'):
        md = MyData(pair['cnt'])
        md.link_to_edf(pair['edf'])
        break

    print(md.raw)
    print(md.edf_raw)
    print(md.epochs)


# %%

# %% ---- 2025-09-29 ------------------------
# Pending

# %% ---- 2025-09-29 ------------------------
# Pending
