"""
File: fbcsp_vote.py
Author: Chuncheng Zhang
Date: 2025-09-25
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Vote on FBCSP results

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
import seaborn as sns
from sklearn import metrics

from util.easy_import import *

# %%
results = joblib.load(
    './results/decoding/fbcsp-results.dump')

output_directory = Path('./results/decoding')
output_directory.mkdir(exist_ok=True, parents=True)

# %% ---- 2025-09-25 ------------------------
# Function and class


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
# Play ground
print('Bands:')

proba_stack = []

for k, v in results.items():
    print(k)
    v['y_pred'] = np.argmax(v['y_proba'], axis=1) + 1
    print(v['mean_accuracy'])
    print(metrics.classification_report(
        y_true=v['y_true'], y_pred=v['y_pred']))
    print(v['all_scores'])

    proba_stack.append(v['y_proba'])

y_true = v['y_true']
print(f'{v.keys()=}')

proba = np.array(proba_stack)
vote_proba = np.prod(proba, axis=0)
y_vote = np.argmax(vote_proba, axis=1) + 1

report = metrics.classification_report(
    y_true=y_true, y_pred=y_vote, output_dict=True)
print(report)

fig, df = plot_fbcsp_results(results)
plt.axhline(y=0.5, color='red', linestyle='--',
            linewidth=1, alpha=0.7, label='y=0.5')
plt.axhline(y=report['accuracy'], color='red', linestyle='--',
            linewidth=1, alpha=0.7, label='vote')
plt.text(x=5, y=report['accuracy'], s='Vote Acc: {:.3f}'.format(report['accuracy']),
         ha='center', va='bottom', fontsize=12, fontweight='bold', fontdict={'color': 'red'})
fig.tight_layout()
fig.savefig(output_directory.joinpath(
    'scores-by-frequency-band-with-vote.png'))
plt.show()


# %% ---- 2025-09-25 ------------------------
# Pending


# %% ---- 2025-09-25 ------------------------
# Pending
