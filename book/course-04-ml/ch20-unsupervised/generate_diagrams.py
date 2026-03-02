#!/usr/bin/env python3
"""Generate all diagrams for Chapter 20: Unsupervised Learning - Clustering"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# Set style
plt.style.use('default')
np.random.seed(42)

print("Generating diagrams for Chapter 20...")

# ============================================================================
# DIAGRAM 1: K-Means Algorithm Steps
# ============================================================================
print("\n1. Creating kmeans_steps.png...")

# Generate sample data with 3 natural clusters
X, y_true = make_blobs(n_samples=150, centers=3, n_features=2,
                       cluster_std=0.8, random_state=42)

# Create figure with 4 panels showing K-Means steps
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('K-Means Algorithm: Step-by-Step', fontsize=16, fontweight='bold')

# Panel 1: Initial random centroids
axes[0, 0].scatter(X[:, 0], X[:, 1], c='#607D8B', alpha=0.6, s=50)
initial_centroids = X[np.random.choice(150, 3, replace=False)]
axes[0, 0].scatter(initial_centroids[:, 0], initial_centroids[:, 1],
                   c='#F44336', marker='*', s=500, edgecolors='black', linewidths=2)
axes[0, 0].set_title('Step 1: Initialize Random Centroids', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Feature 1', fontsize=11)
axes[0, 0].set_ylabel('Feature 2', fontsize=11)

# Panel 2: First assignment
kmeans_iter1 = KMeans(n_clusters=3, init=initial_centroids, n_init=1, max_iter=1, random_state=42)
kmeans_iter1.fit(X)
colors = ['#2196F3', '#4CAF50', '#FF9800']
for i in range(3):
    mask = kmeans_iter1.labels_ == i
    axes[0, 1].scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6, s=50)
axes[0, 1].scatter(kmeans_iter1.cluster_centers_[:, 0], kmeans_iter1.cluster_centers_[:, 1],
                   c='#F44336', marker='*', s=500, edgecolors='black', linewidths=2)
axes[0, 1].set_title('Step 2: Assign Points to Nearest Centroid', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Feature 1', fontsize=11)
axes[0, 1].set_ylabel('Feature 2', fontsize=11)

# Panel 3: Update centroids
kmeans_iter3 = KMeans(n_clusters=3, init=initial_centroids, n_init=1, max_iter=3, random_state=42)
kmeans_iter3.fit(X)
for i in range(3):
    mask = kmeans_iter3.labels_ == i
    axes[1, 0].scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6, s=50)
axes[1, 0].scatter(kmeans_iter3.cluster_centers_[:, 0], kmeans_iter3.cluster_centers_[:, 1],
                   c='#F44336', marker='*', s=500, edgecolors='black', linewidths=2)
axes[1, 0].set_title('Step 3: Update Centroids to Cluster Means', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Feature 1', fontsize=11)
axes[1, 0].set_ylabel('Feature 2', fontsize=11)

# Panel 4: Final convergence
kmeans_final = KMeans(n_clusters=3, random_state=42)
kmeans_final.fit(X)
for i in range(3):
    mask = kmeans_final.labels_ == i
    axes[1, 1].scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6, s=50)
axes[1, 1].scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1],
                   c='#F44336', marker='*', s=500, edgecolors='black', linewidths=2)
axes[1, 1].set_title('Step 4: Converged (Centroids Stable)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Feature 1', fontsize=11)
axes[1, 1].set_ylabel('Feature 2', fontsize=11)

plt.tight_layout()
plt.savefig('diagrams/kmeans_steps.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved kmeans_steps.png")

# ============================================================================
# DIAGRAM 2: K-Means Iris Results
# ============================================================================
print("\n2. Creating kmeans_iris_results.png...")

# Load Iris dataset (using only 2 features for easy visualization)
iris = load_iris()
X = iris.data[:, :2]  # Sepal length and width only
y_true = iris.target
feature_names = iris.feature_names[:2]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, init='k-means++', n_init='auto', max_iter=300, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Visualize
plt.figure(figsize=(14, 6))

# Left panel: Discovered clusters
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis',
                       alpha=0.6, s=100, edgecolors='black', linewidths=0.5)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='#F44336', marker='*',
            s=500, edgecolors='black', linewidths=2, label='Centroids')
plt.xlabel(feature_names[0], fontsize=12)
plt.ylabel(feature_names[1], fontsize=12)
plt.title('K-Means Discovered Clusters (k=3)', fontsize=14, fontweight='bold')
plt.legend()
plt.colorbar(scatter1, label='Cluster')

# Right panel: True species labels
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='plasma',
                       alpha=0.6, s=100, edgecolors='black', linewidths=0.5)
plt.xlabel(feature_names[0], fontsize=12)
plt.ylabel(feature_names[1], fontsize=12)
plt.title('True Iris Species', fontsize=14, fontweight='bold')
plt.colorbar(scatter2, label='Species', ticks=[0, 1, 2])

plt.tight_layout()
plt.savefig('diagrams/kmeans_iris_results.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved kmeans_iris_results.png")

# ============================================================================
# DIAGRAM 3: Optimal K Selection
# ============================================================================
print("\n3. Creating optimal_k_selection.png...")

# Load Iris dataset (all 4 features)
iris = load_iris()
X = iris.data

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test range of k values
k_range = range(2, 11)
wcss_values = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    wcss_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))

# Visualize both metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
axes[0].plot(k_range, wcss_values, 'o-', color='#2196F3', linewidth=2, markersize=10)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0].set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
axes[0].set_title('Elbow Method: Finding Optimal k', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].annotate('Elbow at k=3', xy=(3, wcss_values[1]), xytext=(5, 120),
                arrowprops=dict(arrowstyle='->', color='#F44336', lw=2),
                fontsize=12, color='#F44336', fontweight='bold')

# Silhouette score plot
axes[1].plot(k_range, silhouette_scores, 'o-', color='#4CAF50', linewidth=2, markersize=10)
axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Analysis: Cluster Quality', fontsize=14, fontweight='bold')
axes[1].axhline(y=0.5, color='#F44336', linestyle='--', linewidth=2, label='Good threshold (0.5)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('diagrams/optimal_k_selection.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved optimal_k_selection.png")

# ============================================================================
# DIAGRAM 4 & 5: Hierarchical Clustering
# ============================================================================
print("\n4. Creating hierarchical_dendrogram.png...")

# Load Iris dataset (first 50 samples for readable dendrogram)
iris = load_iris()
X = iris.data[:50]
y_true = iris.target[:50]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create linkage matrix using Ward's method
Z = linkage(X_scaled, method='ward')

# Create dendrogram
plt.figure(figsize=(14, 7))
dendrogram_plot = dendrogram(
    Z,
    truncate_mode=None,
    color_threshold=3.0,
    above_threshold_color='#607D8B'
)

plt.axhline(y=3.0, color='#F44336', linestyle='--', linewidth=2, label='Cut for k=3')
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Distance (Ward Linkage)', fontsize=12)
plt.title('Hierarchical Clustering Dendrogram (First 50 Iris Samples)',
          fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('diagrams/hierarchical_dendrogram.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved hierarchical_dendrogram.png")

print("\n5. Creating hierarchical_vs_kmeans.png...")

# Apply hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Visualize comparison (using first 2 features for 2D plot)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Hierarchical clustering results
for i in range(3):
    mask = hierarchical_labels == i
    axes[0].scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6, s=100, edgecolors='black', linewidths=0.5)
axes[0].set_xlabel('Sepal Length (cm)', fontsize=12)
axes[0].set_ylabel('Sepal Width (cm)', fontsize=12)
axes[0].set_title('Hierarchical Clustering (Ward Linkage)', fontsize=14, fontweight='bold')

# K-Means results
for i in range(3):
    mask = kmeans_labels == i
    axes[1].scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6, s=100, edgecolors='black', linewidths=0.5)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
axes[1].scatter(centers[:, 0], centers[:, 1], c='#F44336', marker='*',
                s=500, edgecolors='black', linewidths=2, label='Centroids')
axes[1].set_xlabel('Sepal Length (cm)', fontsize=12)
axes[1].set_ylabel('Sepal Width (cm)', fontsize=12)
axes[1].set_title('K-Means Clustering', fontsize=14, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('diagrams/hierarchical_vs_kmeans.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved hierarchical_vs_kmeans.png")

# ============================================================================
# DIAGRAM 6 & 7: Customer Segmentation
# ============================================================================
print("\n6. Creating customer_segmentation_k_selection.png...")

# Generate synthetic customer data
np.random.seed(42)
n_customers = 500

# Create 4 natural customer segments
segment_1 = {
    'recency': np.random.normal(10, 5, 125),
    'frequency': np.random.normal(50, 10, 125),
    'monetary': np.random.normal(500, 100, 125)
}
segment_2 = {
    'recency': np.random.normal(40, 10, 125),
    'frequency': np.random.normal(30, 8, 125),
    'monetary': np.random.normal(250, 50, 125)
}
segment_3 = {
    'recency': np.random.normal(180, 30, 125),
    'frequency': np.random.normal(25, 8, 125),
    'monetary': np.random.normal(200, 40, 125)
}
segment_4 = {
    'recency': np.random.normal(300, 40, 125),
    'frequency': np.random.normal(5, 3, 125),
    'monetary': np.random.normal(50, 20, 125)
}

# Combine into DataFrame
df = pd.DataFrame({
    'recency': np.concatenate([segment_1['recency'], segment_2['recency'],
                              segment_3['recency'], segment_4['recency']]),
    'frequency': np.concatenate([segment_1['frequency'], segment_2['frequency'],
                                segment_3['frequency'], segment_4['frequency']]),
    'monetary': np.concatenate([segment_1['monetary'], segment_2['monetary'],
                               segment_3['monetary'], segment_4['monetary']])
})

# Clip to realistic ranges
df['recency'] = df['recency'].clip(lower=0)
df['frequency'] = df['frequency'].clip(lower=1).round()
df['monetary'] = df['monetary'].clip(lower=10)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Find optimal k
k_range = range(2, 9)
wcss_values = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    wcss_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(k_range, wcss_values, 'o-', color='#2196F3', linewidth=2, markersize=10)
axes[0].set_xlabel('Number of Segments (k)', fontsize=12)
axes[0].set_ylabel('WCSS', fontsize=12)
axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouette_scores, 'o-', color='#4CAF50', linewidth=2, markersize=10)
axes[1].set_xlabel('Number of Segments (k)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/customer_segmentation_k_selection.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved customer_segmentation_k_selection.png")

print("\n7. Creating customer_segments_pca.png...")

# Fit final model with k=4
kmeans = KMeans(n_clusters=4, init='k-means++', n_init='auto', random_state=42)
df['segment'] = kmeans.fit_predict(X_scaled)

# Segment names
segment_names = {
    0: "Lost Customers",
    1: "Loyal Regulars",
    2: "At-Risk",
    3: "Champions"
}

# Visualize with PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))

# Define cluster colors
cluster_colors = ['#F44336', '#4CAF50', '#FF9800', '#2196F3']
for i in range(4):
    mask = df['segment'] == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=cluster_colors[i],
               alpha=0.6, s=100, edgecolors='black', linewidths=0.5, label=segment_names[i])

# Add segment labels
for seg_id, name in segment_names.items():
    seg_center = X_pca[df['segment'] == seg_id].mean(axis=0)
    plt.annotate(name, xy=seg_center, fontsize=13, fontweight='bold',
                color='white', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
plt.title('Customer Segments (PCA Visualization)', fontsize=14, fontweight='bold')
plt.legend(loc='best', framealpha=0.9)
plt.tight_layout()
plt.savefig('diagrams/customer_segments_pca.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved customer_segments_pca.png")

print("\n" + "="*60)
print("All diagrams generated successfully!")
print("="*60)
print("Files created:")
print("  1. diagrams/kmeans_steps.png")
print("  2. diagrams/kmeans_iris_results.png")
print("  3. diagrams/optimal_k_selection.png")
print("  4. diagrams/hierarchical_dendrogram.png")
print("  5. diagrams/hierarchical_vs_kmeans.png")
print("  6. diagrams/customer_segmentation_k_selection.png")
print("  7. diagrams/customer_segments_pca.png")
print("="*60)
