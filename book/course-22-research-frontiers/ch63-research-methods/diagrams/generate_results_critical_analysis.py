import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulated results table from a hypothetical paper
results_data = {
    'Method': [
        'Classical Baseline',
        'Deep Learning (2019)',
        'Transformer (2020)',
        'Our Method (no component A)',
        'Our Method (full)'
    ],
    'Dataset 1 Acc': [
        0.752,
        0.831,
        0.867,
        0.881,
        0.894
    ],
    'Dataset 1 Std': [
        0.012,
        0.008,
        np.nan,  # Missing!
        0.007,
        0.006
    ],
    'Dataset 2 Acc': [
        0.681,
        0.779,
        0.812,
        0.829,
        0.841
    ],
    'Dataset 2 Std': [
        0.015,
        0.011,
        np.nan,  # Missing!
        0.009,
        0.008
    ],
    'Year Published': [
        2015,
        2019,
        2020,
        2024,
        2024
    ]
}

df_results = pd.DataFrame(results_data)
baseline_best = df_results.iloc[2]  # Transformer (best non-proposed baseline)

# Visualization of results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, dataset in enumerate(['Dataset 1 Acc', 'Dataset 2 Acc']):
    ax = axes[idx]

    # Bar chart with error bars
    methods = df_results['Method']
    accuracies = df_results[dataset]
    stds = df_results[f'{dataset.split()[0]} {dataset.split()[1]} Std']

    colors = ['gray', 'lightblue', 'lightblue', 'orange', 'darkgreen']
    bars = ax.bar(range(len(methods)), accuracies, color=colors, alpha=0.7, edgecolor='black')

    # Add error bars where available
    for i, (acc, std) in enumerate(zip(accuracies, stds)):
        if not np.isnan(std):
            ax.errorbar(i, acc, yerr=std, fmt='none', color='black', capsize=5, linewidth=2)
        else:
            # Highlight missing error bars
            ax.text(i, acc + 0.01, '⚠️', ha='center', fontsize=12, color='red')

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(dataset, fontsize=12, weight='bold')
    ax.set_ylim([0.6, 0.95])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=baseline_best[dataset], color='red', linestyle='--',
               linewidth=1, alpha=0.5, label='Best baseline')

axes[1].legend()
plt.suptitle('Critical Reading: What Does This Results Table Actually Show?',
             fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-22/ch63/diagrams/results_critical_analysis.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ results_critical_analysis.png saved")
