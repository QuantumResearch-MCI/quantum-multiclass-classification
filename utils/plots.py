import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import math

import math
import matplotlib.pyplot as plt
import seaborn as sns

def plot_conf_matrix(confusion_matrices, ncols=4):
    """
    confusion_matrices : dict {name: cm} from aggregate_results()
                         OR a single np.ndarray (for one model)
    ncols              : number of columns in the subplot grid
    """
    # accept single matrix too
    if not isinstance(confusion_matrices, dict):
        confusion_matrices = {'Confusion Matrix': confusion_matrices}

    n     = len(confusion_matrices)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for i, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(name, fontsize=10)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_performance_comparison(df, all_results, metric='accuracy'):
    # plt.rcParams.update(plt.rcParamsDefault)
    # plt.rcParams.update({
    #     'font.size': 9,
    #     'axes.linewidth': 0.8,
    #     'xtick.major.width': 0.8,
    #     'ytick.major.width': 0.8,
    #     'xtick.major.size': 3,
    #     'ytick.major.size': 3,
    # })

    model_names = [n.replace(' -', '') for n in df.index.tolist()]
    raw_names   = df.index.tolist()
    x = np.arange(len(model_names))

    means = df[f'{metric}_mean'].values
    stds  = df[f'{metric}_std'].values

    fold_values = [
        all_results[name.replace(' ', '_').removesuffix('_-')][metric]
        for name in raw_names
    ]

    BLUE  = '#A8D4E6'   # paper's light blue (classical)
    PURP  = '#B085C8'   # paper's purple (scatter)
    rng   = np.random.default_rng(42)

    fig = plt.figure(figsize=(max(8, len(model_names) * 0.85), 6))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # ax1.set_ylim(0, 1)
    # ax2.set_ylim(0, 1)

    # ── top: bar chart ────────────────────────────────────────────
    ax1.bar(x, means, 0.6,
            color=BLUE, edgecolor='#5AAAC8', linewidth=0.8,
            yerr=stds, capsize=3,
            error_kw=dict(elinewidth=0.9, ecolor='#444', capthick=0.9),
            zorder=2)

    best = max(means)
    ax1.axhline(best, color='gray', lw=0.8, ls='--', alpha=0.6)

    ypad = (means.max() - means.min()) * 0.4
    # ax1.set_ylim(max(0, means.min() - ypad), means.max() + ypad * 0.6)
    ax1.set_ylabel('Mean Accuracy', fontsize=9)
    ax1.yaxis.grid(True, ls='--', lw=0.5, alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)
    ax1.tick_params(labelbottom=False, labelsize=8)
    for spine in ax1.spines.values():
        spine.set_linewidth(0.8)

    # ── bottom: per-fold scatter ──────────────────────────────────
    all_fold_vals = [v for folds in fold_values for v in folds]
    ypad_sc = (max(all_fold_vals) - min(all_fold_vals)) * 0.3
    # ax2.set_ylim(max(0, min(all_fold_vals) - ypad_sc),
    #              min(1.0, max(all_fold_vals) + ypad_sc * 0.5))

    for i, folds in enumerate(fold_values):
        folds = np.array(folds)
        jitter = rng.uniform(-0.18, 0.18, len(folds))

        # range line + caps
        ax2.vlines(i, folds.min(), folds.max(),
                   color=PURP, lw=0.9, zorder=2)
        ax2.hlines([folds.min(), folds.max()],
                   i - 0.09, i + 0.09,
                   color=PURP, lw=0.9, zorder=2)

        # faded large dots (paper style)
        ax2.scatter(i + jitter, folds,
                    color=PURP, s=55, alpha=0.45, zorder=3,
                    linewidths=0, edgecolors='none')

    ax2.axhline(best, color='gray', lw=0.8, ls='--', alpha=0.6)
    ax2.set_ylabel('Accuracy', fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=25, ha='right', fontsize=8)
    ax2.yaxis.grid(True, ls='--', lw=0.5, alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)
    ax2.tick_params(labelsize=8)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.8)

    fig.suptitle(f'Performance Comparison  —  {metric}', fontsize=10, y=1.01)
    plt.tight_layout()
    plt.show()

    # return fig

def plot_roc_curve(all_results, model_names, le, ncols=4):
    """
    all_results  : raw results dict
    model_names  : single string OR list of strings e.g. 'SVC_linear' or ['SVC_linear', 'XGB']
    le           : the LabelEncoder used during training
    """
    # accept both single string and list
    if isinstance(model_names, str):
        model_names = [model_names]

    n      = len(model_names)
    ncols  = min(ncols, n)
    nrows  = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    classes   = le.classes_
    n_classes = len(classes)

    for ax, model_name in zip(axes, model_names):
        key        = model_name.replace(' ', '_').removesuffix('_-')
        y_true_all = np.concatenate(all_results[key]['y_true'])
        y_prob_all = np.concatenate(all_results[key]['y_prob'], axis=0)

        y_bin = label_binarize(y_true_all, classes=np.arange(n_classes))

        fpr, tpr, roc_auc = {}, {}, {}

        # per-class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob_all[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # micro
        fpr['micro'], tpr['micro'], _ = roc_curve(y_bin.ravel(), y_prob_all.ravel())
        roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

        # macro
        all_fpr  = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr['macro'], tpr['macro'] = all_fpr, mean_tpr
        roc_auc['macro'] = auc(all_fpr, mean_tpr)

        # weighted
        class_counts = np.bincount(y_true_all, minlength=n_classes)
        weights      = class_counts / class_counts.sum()
        all_fpr_w    = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr_w   = np.zeros_like(all_fpr_w)
        for i in range(n_classes):
            mean_tpr_w += weights[i] * np.interp(all_fpr_w, fpr[i], tpr[i])
        fpr['weighted'], tpr['weighted'] = all_fpr_w, mean_tpr_w
        roc_auc['weighted'] = auc(all_fpr_w, mean_tpr_w)

        # ── draw ──────────────────────────────────────────────
        colors = cycle(plt.cm.tab10.colors)
        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=1.5,
                    label=f'Class {classes[i]} (AUC = {roc_auc[i]:.4f})')

        ax.plot(fpr['macro'],    tpr['macro'],    color='grey',   lw=1.5, ls='--',
                label=f'Macroaverage (AUC = {roc_auc["macro"]:.4f})')
        ax.plot(fpr['micro'],    tpr['micro'],    color='navy',   lw=1.5, ls=':',
                label=f'Microaverage (AUC = {roc_auc["micro"]:.4f})')
        ax.plot(fpr['weighted'], tpr['weighted'], color='purple', lw=1.5, ls='-.',
                label=f'Weighted average (AUC = {roc_auc["weighted"]:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=9)
        ax.set_ylabel('True Positive Rate',  fontsize=9)
        ax.set_title(model_name, fontsize=10)
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(True, ls='--', alpha=0.3)

    # hide unused axes
    for j in range(len(model_names), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()