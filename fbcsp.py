"""
File: fbcsp.py
Author: Chuncheng Zhang
Date: 2025-09-25
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    FBCSP decoding.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-09-25 ------------------------
# Requirements and constants
import joblib
import mne
import seaborn as sns
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LeaveOneGroupOut

from util.easy_import import *
from read_data import MyData

# %%
raw_directory = Path('./raw')
cnt_files = sorted(list(raw_directory.rglob('*.cnt')))
print(f'{cnt_files=}')

output_directory = Path('./results/decoding-customized-channels')
output_directory.mkdir(exist_ok=True, parents=True)

customized_ch_names = open('./results/ch_names.txt').read().split()

# %% ---- 2025-09-25 ------------------------
# Function and class


def create_fbcsp_pipeline(freq_bands=None, n_components=4):
    """
    创建FBCSP管道

    参数:
    freq_bands: 频率带列表，每个元素为(low_freq, high_freq)
    n_components: CSP组件数量
    """
    if freq_bands is None:
        # 默认频率带（根据你的任务调整）
        freq_bands = [(8, 12), (12, 16), (16, 20),
                      (20, 24), (24, 28), (28, 32)]

    pipelines = []

    for i, (low_freq, high_freq) in enumerate(freq_bands):
        # 为每个频率带创建CSP + LDA管道
        # csp = CSP(n_components=n_components, reg=None, log=True,
        #           norm_trace=False, transform_into='average_power')

        # 创建管道 - 关键：CSP在管道内部，会在每个CV fold中独立拟合
        pipeline = Pipeline([
            (f'csp_{i}', CSP(n_components=n_components, reg=None, log=True,
                             norm_trace=False, transform_into='average_power')),
            (f'lda_{i}', LinearDiscriminantAnalysis())
        ])
        pipelines.append(
            (f'band_{low_freq}_{high_freq}Hz', pipeline, low_freq, high_freq))

    return pipelines, freq_bands


def evaluate_fbcsp_group_cv(X, y, groups, pipelines):
    """
    使用Group K-Fold交叉验证评估FBCSP

    参数:
    X: EEG数据 (n_samples, n_channels, n_times)
    y: 标签 (n_samples,)
    groups: 分组信息 (n_samples,)
    pipelines: FBCSP管道列表
    """

    # 创建Group K-Fold
    # group_kfold = GroupKFold(n_splits=cv_folds)
    cv = LeaveOneGroupOut()

    results = {}

    for name, pipeline, low_freq, high_freq in pipelines:
        print(f"\n正在处理频率带: {low_freq}-{high_freq} Hz")

        # 频带滤波
        X_filtered = X.copy().astype(np.float64, copy=False)
        for i in range(X.shape[0]):
            # 创建Epochs对象进行滤波
            _epochs = mne.EpochsArray(X[i:i+1], epochs.info)
            epochs_filtered = _epochs.filter(l_freq=low_freq, h_freq=high_freq,
                                             method='iir', verbose=False)
            X_filtered[i] = epochs_filtered.get_data()[0]

        # 交叉验证
        cv_scores = cross_val_score(pipeline, X_filtered, y,
                                    cv=cv, groups=groups,
                                    scoring='accuracy', n_jobs=1)

        y_proba = cross_val_predict(pipeline, X_filtered, y,
                                    cv=cv, groups=groups, method='predict_proba')

        results[name] = {
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'all_scores': cv_scores,
            'low_freq': low_freq,
            'high_freq': high_freq,
            'y_true': y,
            'y_proba': y_proba
        }

        print(
            f"频率带 {low_freq}-{high_freq} Hz 准确率: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    return results


def plot_fbcsp_results(results):
    """用seaborn绘制美观的FBCSP结果条形图"""

    # 准备数据
    bands = []
    accuracies = []
    errors = []

    for band_name, result in results.items():
        bands.append(f"{result['low_freq']}-{result['high_freq']}Hz")
        accuracies.append(result['mean_accuracy'])
        errors.append(result['std_accuracy'])

    # 创建DataFrame
    df = pd.DataFrame({
        'Frequency Band': bands,
        'Accuracy': accuracies,
        'Error': errors
    })

    # 创建图形
    fig = plt.figure(figsize=(12, 6))

    # 使用seaborn绘制条形图
    ax = sns.barplot(data=df, x='Frequency Band', y='Accuracy', hue='Frequency Band',
                     palette='viridis', alpha=0.8, edgecolor='black', linewidth=0.5)

    # 添加误差线
    x_coords = [patch.get_x() + patch.get_width() / 2 for patch in ax.patches]
    ax.errorbar(x=x_coords, y=accuracies, yerr=errors,
                fmt='none', c='black', capsize=5, capthick=1, linewidth=1.5)

    # 在柱子上添加数值标签
    for i, (acc, err) in enumerate(zip(accuracies, errors)):
        ax.text(i, acc + 0.01, f'{acc:.3f}\n±{err:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 美化图形
    ax.set_xlabel('Frequency Bands (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('FBCSP Performance by Frequency Band',
                 fontsize=14, fontweight='bold', pad=20)

    # 旋转x轴标签避免重叠
    plt.xticks(rotation=45, ha='right')

    # 设置y轴范围，留出空间显示标签
    y_min = min(accuracies) - 0.1
    y_max = max(accuracies) + 0.15
    ax.set_ylim(max(0, y_min), min(1.0, y_max))

    # 添加网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(False)

    # 移除右边和上边的边框
    sns.despine(top=True, right=True)

    return fig, df


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
epochs.pick(customized_ch_names)
epochs

# Generate X, y
X = epochs.get_data(copy=False)
y = epochs.events[:, -1]


# %% ---- 2025-09-25 ------------------------
# Pending

# 创建FBCSP管道
pipelines, freq_bands = create_fbcsp_pipeline()

# 运行评估
results = evaluate_fbcsp_group_cv(X, y, groups, pipelines)

# %%
# 绘制结果
fig, results_df = plot_fbcsp_results(results)
plt.tight_layout()
# plt.show()
fig.savefig(output_directory.joinpath('scores-by-frequency-band.png'))

joblib.dump(results, output_directory.joinpath('fbcsp-results.dump'))

# 打印结果
print(results_df)

# %% ---- 2025-09-25 ------------------------
# Pending

# %%
# %%

# %%
