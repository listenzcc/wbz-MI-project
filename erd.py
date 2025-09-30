"""
File: erd.py
Author: Chuncheng Zhang
Date: 2025-09-25
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read data and ERD analysis.

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
from mne.stats import permutation_cluster_1samp_test as pcluster_test

from util.easy_import import *
from read_data_link import MyData


# %%
DATA_DIR = Path('./raw/20250929')
OUTPUT_DIR = Path('./results/erd')
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

# %% ---- 2025-09-25 ------------------------
# Function and class


# %% ---- 2025-09-25 ------------------------
# Play ground
raw_epochs = mne.concatenate_epochs([md.epochs for md in mds])
raw_epochs.load_data()

epochs = raw_epochs.copy().pick(['C3', 'CZ', 'C4'])
event_ids = list(epochs.event_id.keys())


# %%
for evt in event_ids:
    evoked = raw_epochs[evt].average()
    fig = evoked.plot_joint(
        show=False, title=f'{evt=}', times=[0, 0.135, 0.470, 1, 2])
    fig.savefig(OUTPUT_DIR.parent.joinpath(f'evoked-{evt=}.png'))
    # plt.show()

# %%

freqs = np.arange(2, 36)  # frequencies from 2-35Hz
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
baseline = (-1, 0)  # baseline interval (in s)
cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

kwargs = dict(
    n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask"
)  # for cluster test

tfr = epochs.compute_tfr(
    method="multitaper",
    freqs=freqs,
    n_cycles=freqs,
    use_fft=True,
    return_itc=False,
    average=False,
    decim=2,
)
tfr.apply_baseline(baseline, mode="percent")

for evt in event_ids:
    # select desired epochs for visualization
    tfr_ev = tfr[evt]
    fig, axes = plt.subplots(
        1, 4, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 10, 1]}
    )
    for ch, ax in enumerate(axes[:-1]):  # for each channel
        # positive clusters
        _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
        # negative clusters
        _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)

        # note that we keep clusters with p <= 0.05 from the combined clusters
        # of two independent tests; in this example, we do not correct for
        # these two comparisons
        c = np.stack(c1 + c2, axis=2)  # combined clusters
        p = np.concatenate((p1, p2))  # combined p-values
        mask = c[..., p <= 0.05].any(axis=-1)

        # plot TFR (ERDS map with masking)
        tfr_ev.average().plot(
            [ch],
            cmap="RdBu",
            cnorm=cnorm,
            axes=ax,
            colorbar=False,
            show=False,
            mask=mask,
            mask_style="mask",
        )

        ax.set_title(epochs.ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if ch != 0:
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
    fig.suptitle(f"ERDS ({evt})")
    fig.savefig(OUTPUT_DIR.joinpath(f'ERDS-{evt=}.png'))
    # plt.show()

# %% ---- 2025-09-25 ------------------------
# Pending


# %% ---- 2025-09-25 ------------------------
# Pending
