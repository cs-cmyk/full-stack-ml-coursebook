#!/usr/bin/env python3
"""
Generate all diagrams for Chapter 35: Recommender Systems
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Create output directory
output_dir = 'diagrams'
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

print("Generating diagrams for Chapter 35: Recommender Systems...")

# =============================================================================
# DIAGRAM 1: User-Item Matrix and Matrix Factorization
# =============================================================================
print("1. Generating user-item matrix and matrix factorization diagram...")

# Create a synthetic sparse user-item rating matrix
n_users = 20
n_items = 25
density = 0.15  # 15% of entries have ratings (85% sparse)

# Generate sparse ratings
ratings_matrix = np.full((n_users, n_items), np.nan)
n_ratings = int(n_users * n_items * density)
user_indices = np.random.choice(n_users, size=n_ratings, replace=True)
item_indices = np.random.choice(n_items, size=n_ratings, replace=True)
rating_values = np.random.randint(1, 6, size=n_ratings)

for u, i, r in zip(user_indices, item_indices, rating_values):
    ratings_matrix[u, i] = r

# Visualize the sparse matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Heatmap showing ratings and sparsity
mask = np.isnan(ratings_matrix)
ax1 = axes[0]
sns.heatmap(ratings_matrix, mask=~mask, cmap='YlOrRd', cbar_kws={'label': 'Rating'},
            ax=ax1, vmin=1, vmax=5, linewidths=0.1, linecolor='gray')
sns.heatmap(ratings_matrix, mask=mask, cmap=['lightgray'], cbar=False, ax=ax1,
            linewidths=0.1, linecolor='gray')
ax1.set_xlabel('Items (Movies)', fontsize=12)
ax1.set_ylabel('Users', fontsize=12)
ax1.set_title(f'User-Item Rating Matrix\n({100*density:.0f}% observed, {100*(1-density):.0f}% missing)',
              fontsize=13, fontweight='bold')

# Right: Matrix factorization visualization
ax2 = axes[1]
# Show conceptual matrix factorization
# R (n×m) ≈ U (n×k) × V^T (k×m)
n, m, k = 20, 25, 5

# Draw rectangles for matrices
rect_R = Rectangle((0, 0), 3, 2, linewidth=2, edgecolor='black', facecolor='lightcoral', alpha=0.5)
rect_U = Rectangle((5, 0), 1.5, 2, linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.5)
rect_V = Rectangle((7, 0.5), 3, 1, linewidth=2, edgecolor='black', facecolor='lightgreen', alpha=0.5)

ax2.add_patch(rect_R)
ax2.add_patch(rect_U)
ax2.add_patch(rect_V)

# Add labels
ax2.text(1.5, 1, 'R', fontsize=24, ha='center', va='center', fontweight='bold')
ax2.text(1.5, -0.5, f'{n} users × {m} items', fontsize=11, ha='center')
ax2.text(5.75, 1, 'U', fontsize=24, ha='center', va='center', fontweight='bold')
ax2.text(5.75, -0.5, f'{n} × {k}', fontsize=11, ha='center')
ax2.text(8.5, 1, 'V^T', fontsize=24, ha='center', va='center', fontweight='bold')
ax2.text(8.5, 1.8, f'{k} × {m}', fontsize=11, ha='center')

# Add approximation symbol and multiplication
ax2.text(3.8, 1, '≈', fontsize=28, ha='center', va='center')
ax2.text(6.7, 1, '×', fontsize=28, ha='center', va='center')

# Annotations
ax2.annotate('User factors\n(preferences)', xy=(5.75, 2), xytext=(5.75, 3),
            fontsize=10, ha='center', arrowprops=dict(arrowstyle='->', lw=1.5))
ax2.annotate('Item factors\n(characteristics)', xy=(8.5, 0.5), xytext=(8.5, -1.2),
            fontsize=10, ha='center', arrowprops=dict(arrowstyle='->', lw=1.5))
ax2.annotate(f'{k} latent\nfactors', xy=(6.5, 1.7), fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

ax2.set_xlim(-0.5, 11)
ax2.set_ylim(-1.5, 3.5)
ax2.axis('off')
ax2.set_title('Matrix Factorization Concept\nDecomposing sparse R into low-rank U and V',
              fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/user_item_matrix_and_factorization.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("   ✓ Saved: user_item_matrix_and_factorization.png")

# =============================================================================
# DIAGRAM 2: SVD Hyperparameter Tuning
# =============================================================================
print("2. Generating SVD hyperparameter tuning diagram...")

# Simulated hyperparameter tuning results
n_factors_range = [5, 10, 20, 30, 40, 50]
reg_values = [0.01, 0.02, 0.05]

# Simulate RMSE values (lower is better, with optimal around 20-30 factors)
rmse_results = {}
for reg in reg_values:
    rmse_results[reg] = []
    for k in n_factors_range:
        # Simulate U-shaped curve with minimum around k=20-30
        base_rmse = 0.95 + 0.15 * (k - 25)**2 / 625  # U-shaped
        noise = np.random.normal(0, 0.02)
        rmse = base_rmse + reg * 5 + noise  # Higher regularization = higher RMSE
        rmse_results[reg].append(rmse)

fig, ax = plt.subplots(figsize=(8, 6))

colors = ['#2196F3', '#4CAF50', '#FF9800']
markers = ['o', 's', '^']

for i, reg in enumerate(reg_values):
    ax.plot(n_factors_range, rmse_results[reg], marker=markers[i],
            color=colors[i], linewidth=2, markersize=8, label=f'λ = {reg}')

ax.set_xlabel('Number of Latent Factors (k)', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax.set_title('Hyperparameter Tuning: RMSE vs. Number of Factors', fontsize=13, fontweight='bold')
ax.legend(title='Regularization', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/svd_hyperparameter_tuning.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("   ✓ Saved: svd_hyperparameter_tuning.png")

# =============================================================================
# DIAGRAM 3: Ranking Metrics Comparison
# =============================================================================
print("3. Generating ranking metrics comparison diagram...")

# Simulated ranking metrics
k_values = [1, 3, 5, 10, 20, 30]

# Precision@K (decreases with K) and Recall@K (increases with K)
precision_item_cf = [0.85, 0.75, 0.68, 0.55, 0.42, 0.35]
recall_item_cf = [0.12, 0.28, 0.38, 0.52, 0.65, 0.72]
precision_svd = [0.88, 0.78, 0.72, 0.60, 0.48, 0.40]
recall_svd = [0.15, 0.32, 0.42, 0.58, 0.70, 0.78]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Precision@K and Recall@K vs K
ax1 = axes[0]
ax1.plot(k_values, precision_item_cf, marker='o', linestyle='-', linewidth=2,
         color='#2196F3', label='Precision@K (Item-CF)', markersize=8)
ax1.plot(k_values, recall_item_cf, marker='s', linestyle='--', linewidth=2,
         color='#2196F3', label='Recall@K (Item-CF)', markersize=8)
ax1.plot(k_values, precision_svd, marker='o', linestyle='-', linewidth=2,
         color='#4CAF50', label='Precision@K (SVD)', markersize=8)
ax1.plot(k_values, recall_svd, marker='s', linestyle='--', linewidth=2,
         color='#4CAF50', label='Recall@K (SVD)', markersize=8)

ax1.set_xlabel('K (Number of Recommendations)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Precision@K and Recall@K vs. K', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Right: Precision-Recall tradeoff
ax2 = axes[1]
ax2.plot(recall_item_cf, precision_item_cf, marker='o', linestyle='-', linewidth=2,
         color='#2196F3', label='Item-Based CF', markersize=8)
ax2.plot(recall_svd, precision_svd, marker='s', linestyle='-', linewidth=2,
         color='#4CAF50', label='SVD', markersize=8)

# Annotate some K values
for i, k in enumerate([1, 10, 30]):
    idx = k_values.index(k)
    ax2.annotate(f'K={k}', xy=(recall_item_cf[idx], precision_item_cf[idx]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax2.set_xlabel('Recall@K', fontsize=12, fontweight='bold')
ax2.set_ylabel('Precision@K', fontsize=12, fontweight='bold')
ax2.set_title('Precision-Recall Tradeoff', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/ranking_metrics_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("   ✓ Saved: ranking_metrics_comparison.png")

# =============================================================================
# DIAGRAM 4: Neural CF Training and Model Comparison
# =============================================================================
print("4. Generating Neural CF training and comparison diagram...")

# Simulated training loss
epochs = np.arange(1, 21)
training_loss = 1.2 * np.exp(-epochs / 5) + 0.3 + np.random.normal(0, 0.02, len(epochs))

# Model comparison metrics
models = ['Item-Based CF', 'Matrix\nFactorization', 'Neural CF']
rmse_scores = [0.92, 0.88, 0.85]
mae_scores = [0.72, 0.68, 0.65]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Training loss curve
ax1 = axes[0]
ax1.plot(epochs, training_loss, marker='o', linestyle='-', linewidth=2,
         color='#2196F3', markersize=6)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Training Loss (MSE)', fontsize=12, fontweight='bold')
ax1.set_title('Neural CF Training Progress', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Right: Model comparison
ax2 = axes[1]
x = np.arange(len(models))
width = 0.35

bars1 = ax2.bar(x - width/2, rmse_scores, width, label='RMSE', color='#4CAF50', alpha=0.8)
bars2 = ax2.bar(x + width/2, mae_scores, width, label='MAE', color='#FF9800', alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)

ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/ncf_training_and_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("   ✓ Saved: ncf_training_and_comparison.png")

# =============================================================================
# DIAGRAM 5: Hyperparameter Optimization Comprehensive
# =============================================================================
print("5. Generating hyperparameter optimization comprehensive diagram...")

# Create comprehensive hyperparameter analysis
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Top-left: RMSE vs factors for different regularizations
ax1 = fig.add_subplot(gs[0, 0])
n_factors = np.array([5, 10, 20, 30, 40, 50, 75, 100])
reg_values = [0.001, 0.005, 0.01, 0.02, 0.05]
colors_reg = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

for i, reg in enumerate(reg_values):
    # Simulate U-shaped curves with different optima
    rmse = 1.0 + 0.15 * (n_factors - 30)**2 / 900 + reg * 8
    rmse += np.random.normal(0, 0.01, len(n_factors))
    ax1.plot(n_factors, rmse, marker='o', linewidth=2,
             color=colors_reg[i], label=f'λ={reg}', markersize=6)

ax1.set_xlabel('Number of Latent Factors', fontsize=12, fontweight='bold')
ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax1.set_title('RMSE vs. Factors (learning_rate=0.005)', fontsize=13, fontweight='bold')
ax1.legend(title='Regularization', fontsize=9)
ax1.grid(True, alpha=0.3)

# Top-right: Heatmap of RMSE for reg vs learning rate
ax2 = fig.add_subplot(gs[0, 1])
learning_rates = [0.001, 0.003, 0.005, 0.01, 0.02]
reg_for_heatmap = [0.001, 0.005, 0.01, 0.02, 0.05]
rmse_grid = np.zeros((len(reg_for_heatmap), len(learning_rates)))

for i, reg in enumerate(reg_for_heatmap):
    for j, lr in enumerate(learning_rates):
        # Simulate interaction: too high or too low lr is bad
        lr_penalty = 0.1 * (np.log(lr) + 6)**2 / 4
        reg_penalty = reg * 5
        rmse_grid[i, j] = 0.85 + lr_penalty + reg_penalty + np.random.normal(0, 0.01)

sns.heatmap(rmse_grid, annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=[f'{lr}' for lr in learning_rates],
            yticklabels=[f'{r}' for r in reg_for_heatmap],
            cbar_kws={'label': 'RMSE'}, ax=ax2)
ax2.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
ax2.set_ylabel('Regularization (λ)', fontsize=12, fontweight='bold')
ax2.set_title('RMSE Heatmap (k=50)', fontsize=13, fontweight='bold')

# Bottom-left: Optimal factors with overfitting analysis
ax3 = fig.add_subplot(gs[1, 0])
for i, reg in enumerate(reg_values):
    rmse = 1.0 + 0.15 * (n_factors - 30)**2 / 900 + reg * 8
    rmse += np.random.normal(0, 0.01, len(n_factors))
    ax3.plot(n_factors, rmse, marker='o', linewidth=2,
             color=colors_reg[i], label=f'λ={reg}', markersize=6)

# Mark optimal point
optimal_k = 30
ax3.axvline(optimal_k, color='red', linestyle='--', linewidth=2, label='Optimal k')
ax3.set_xlabel('Number of Latent Factors', fontsize=12, fontweight='bold')
ax3.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax3.set_title('Identifying Optimal k (Overfitting Analysis)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Bottom-right: Top configurations
ax4 = fig.add_subplot(gs[1, 1])
top_configs = [
    'k=30, λ=0.005, lr=0.005',
    'k=40, λ=0.001, lr=0.005',
    'k=20, λ=0.01, lr=0.003',
    'k=50, λ=0.005, lr=0.003',
    'k=30, λ=0.001, lr=0.01'
]
top_rmse = [0.856, 0.863, 0.868, 0.871, 0.875]

bars = ax4.barh(range(len(top_configs)), top_rmse, color=colors_reg[:5], alpha=0.8)
ax4.set_yticks(range(len(top_configs)))
ax4.set_yticklabels([f'{i+1}. {cfg}' for i, cfg in enumerate(top_configs)], fontsize=10)
ax4.set_xlabel('RMSE', fontsize=12, fontweight='bold')
ax4.set_title('Top 5 Configurations', fontsize=13, fontweight='bold')
ax4.grid(True, axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_rmse)):
    ax4.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/hyperparameter_optimization_comprehensive.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("   ✓ Saved: hyperparameter_optimization_comprehensive.png")

# =============================================================================
# DIAGRAM 6: Cold Start Analysis
# =============================================================================
print("6. Generating cold start analysis diagram...")

# User groups and metrics
user_groups = ['Cold\n(<5 ratings)', 'Warm\n(5-20 ratings)', 'Hot\n(>20 ratings)']
rmse_by_group = [1.15, 0.92, 0.78]
precision_by_group = [0.32, 0.58, 0.72]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: RMSE by user group
ax1 = axes[0]
colors_groups = ['#F44336', '#FF9800', '#4CAF50']
bars1 = ax1.bar(user_groups, rmse_by_group, color=colors_groups, alpha=0.8, edgecolor='black')

for bar, val in zip(bars1, rmse_by_group):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax1.set_xlabel('User Group', fontsize=12, fontweight='bold')
ax1.set_title('RMSE by User Group', fontsize=13, fontweight='bold')
ax1.set_ylim(0, max(rmse_by_group) * 1.2)
ax1.grid(True, axis='y', alpha=0.3)

# Right: Precision@10 by user group
ax2 = axes[1]
bars2 = ax2.bar(user_groups, precision_by_group, color=colors_groups, alpha=0.8, edgecolor='black')

for bar, val in zip(bars2, precision_by_group):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.2f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax2.set_ylabel('Precision@10', fontsize=12, fontweight='bold')
ax2.set_xlabel('User Group', fontsize=12, fontweight='bold')
ax2.set_title('Precision@10 by User Group', fontsize=13, fontweight='bold')
ax2.set_ylim(0, max(precision_by_group) * 1.2)
ax2.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/cold_start_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("   ✓ Saved: cold_start_analysis.png")

# =============================================================================
# DIAGRAM 7: Similarity Metrics Comparison
# =============================================================================
print("7. Generating similarity metrics comparison diagram...")

# Similarity metrics comparison
metrics_names = ['Cosine', 'Pearson', 'Jaccard', 'Euclidean']
rmse_scores = [0.88, 0.86, 0.95, 0.92]
mae_scores = [0.68, 0.66, 0.74, 0.71]
precision_scores = [0.62, 0.65, 0.55, 0.58]
ndcg_scores = [0.71, 0.73, 0.63, 0.67]

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Top-left: RMSE and MAE comparison
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax1.bar(x - width/2, rmse_scores, width, label='RMSE', color='#2196F3', alpha=0.8)
bars2 = ax1.bar(x + width/2, mae_scores, width, label='MAE', color='#4CAF50', alpha=0.8)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)

ax1.set_xlabel('Similarity Metric', fontsize=12, fontweight='bold')
ax1.set_ylabel('Error Score', fontsize=12, fontweight='bold')
ax1.set_title('Error Metrics by Similarity Function', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_names)
ax1.legend(fontsize=10)
ax1.grid(True, axis='y', alpha=0.3)

# Top-right: Precision and NDCG comparison
ax2 = fig.add_subplot(gs[0, 1])
bars3 = ax2.bar(x - width/2, precision_scores, width, label='Precision@10',
                color='#FF9800', alpha=0.8)
bars4 = ax2.bar(x + width/2, ndcg_scores, width, label='NDCG@10',
                color='#9C27B0', alpha=0.8)

for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)

ax2.set_xlabel('Similarity Metric', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('Ranking Metrics by Similarity Function', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics_names)
ax2.legend(fontsize=10)
ax2.grid(True, axis='y', alpha=0.3)

# Bottom-left: Radar chart
ax3 = fig.add_subplot(gs[1, 0], projection='polar')

# Normalize scores (invert RMSE and MAE so higher is better)
rmse_norm = [1 - (r - min(rmse_scores)) / (max(rmse_scores) - min(rmse_scores)) for r in rmse_scores]
mae_norm = [1 - (m - min(mae_scores)) / (max(mae_scores) - min(mae_scores)) for m in mae_scores]
precision_norm = [(p - min(precision_scores)) / (max(precision_scores) - min(precision_scores)) for p in precision_scores]
ndcg_norm = [(n - min(ndcg_scores)) / (max(ndcg_scores) - min(ndcg_scores)) for n in ndcg_scores]

angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
angles += angles[:1]

colors_radar = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
for i, (name, rmse_n, mae_n, prec_n, ndcg_n, color) in enumerate(zip(
        metrics_names, rmse_norm, mae_norm, precision_norm, ndcg_norm, colors_radar)):
    values = [rmse_n, mae_n, prec_n, ndcg_n]
    values += values[:1]
    ax3.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
    ax3.fill(angles, values, alpha=0.15, color=color)

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(['RMSE', 'MAE', 'Precision@10', 'NDCG@10'], fontsize=10)
ax3.set_ylim(0, 1)
ax3.set_title('Normalized Performance Comparison\n(Larger is Better)',
              fontsize=13, fontweight='bold', pad=20)
ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
ax3.grid(True)

# Bottom-right: Analysis text
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

analysis_text = """
Similarity Metric Analysis:

• Pearson Correlation performs best overall
  - Lowest RMSE (0.86) and MAE (0.66)
  - Highest Precision@10 (0.65) and NDCG@10 (0.73)
  - Accounts for user rating bias

• Cosine Similarity is a close second
  - Strong performance across all metrics
  - Computationally efficient
  - Good for sparse datasets

• Jaccard shows weaker performance
  - Highest errors (RMSE=0.95, MAE=0.74)
  - Only considers binary overlap
  - Ignores rating magnitudes

• Euclidean Distance is moderate
  - Better than Jaccard but worse than correlation
  - Sensitive to rating scale differences

Recommendation: Use Pearson correlation for
rating prediction tasks and Cosine similarity
when computational efficiency is critical.
"""

ax4.text(0.1, 0.95, analysis_text, transform=ax4.transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{output_dir}/similarity_metrics_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("   ✓ Saved: similarity_metrics_comparison.png")

print("\n" + "="*70)
print("✓ All diagrams generated successfully!")
print(f"✓ Saved to: {output_dir}/")
print("="*70)
