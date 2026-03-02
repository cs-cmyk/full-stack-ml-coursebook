# Chapter 20 Diagrams - Unsupervised Learning

All diagrams for Chapter 20: Unsupervised Learning - Clustering

## Generated Diagrams

### 1. kmeans_steps.png (242 KB)
**Purpose:** Illustrate the K-Means algorithm step-by-step
- Panel 1: Initial random centroids
- Panel 2: First assignment of points to nearest centroid
- Panel 3: Updated centroids after iteration
- Panel 4: Final converged state
- **Color Palette:** Blue (#2196F3), Green (#4CAF50), Orange (#FF9800), Red (#F44336) for centroids, Gray (#607D8B) for initial points
- **Referenced in:** Visual section (lines 44-108)

### 2. kmeans_iris_results.png (221 KB)
**Purpose:** Show K-Means clustering results on the Iris dataset
- Left panel: K-Means discovered clusters with centroids
- Right panel: True Iris species labels for comparison
- Demonstrates how well unsupervised learning discovers natural groupings
- **Referenced in:** Code Example section (lines 114-233)

### 3. optimal_k_selection.png (124 KB)
**Purpose:** Guide selection of optimal number of clusters
- Left panel: Elbow plot showing WCSS vs. k
- Right panel: Silhouette score vs. k
- Shows elbow at k=3, indicating optimal cluster count
- **Referenced in:** Finding Optimal K section (lines 263-354)

### 4. hierarchical_dendrogram.png (64 KB)
**Purpose:** Visualize hierarchical clustering tree structure
- Dendrogram showing hierarchical relationships between samples
- Red dashed line showing cut point for k=3 clusters
- Y-axis represents Ward linkage distance
- **Referenced in:** Hierarchical Clustering section (lines 364-490)

### 5. hierarchical_vs_kmeans.png (91 KB)
**Purpose:** Compare hierarchical and K-Means clustering approaches
- Left panel: Hierarchical clustering (Ward linkage) results
- Right panel: K-Means clustering results with centroids
- Shows similar results from both methods
- **Referenced in:** Hierarchical Clustering section (lines 364-490)

### 6. customer_segmentation_k_selection.png (87 KB)
**Purpose:** Optimal k selection for customer segmentation
- Left panel: Elbow method for RFM customer data
- Right panel: Silhouette scores for different k values
- Guides choice of k=4 for customer segments
- **Referenced in:** Real-World Application section (lines 500-712)

### 7. customer_segments_pca.png (301 KB)
**Purpose:** Visualize customer segments in 2D using PCA
- Four distinct customer segments: Champions, Loyal Regulars, At-Risk, Lost Customers
- Each segment labeled with business-friendly names
- PCA reduces 3D RFM space to 2D for visualization
- Shows clear separation between customer groups
- **Referenced in:** Real-World Application section (lines 500-712)

## Technical Specifications

- **Resolution:** 150 DPI (optimized for digital viewing and printing)
- **Format:** PNG with white background
- **Max Width:** ~800px (appropriate for textbook layout)
- **Color Palette:** Consistent educational colors
  - Blue (#2196F3) - Primary
  - Green (#4CAF50) - Secondary
  - Orange (#FF9800) - Tertiary
  - Red (#F44336) - Emphasis/Centroids
  - Purple (#9C27B0) - Alternative
  - Gray (#607D8B) - Neutral
- **Font Sizes:** Minimum 11pt for readability
- **Style:** matplotlib default with tight_layout() for optimal spacing

## Regeneration

To regenerate all diagrams:
```bash
cd /home/chirag/ds-book/book/course-04-ml/ch20-unsupervised
python generate_diagrams.py
```

All diagrams are deterministic (seeded with `random_state=42`) and will produce identical results on each run.
