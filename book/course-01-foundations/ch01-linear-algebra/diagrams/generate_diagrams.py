"""
Generate all diagrams for Chapter 1: Linear Algebra
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# Set consistent style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Color palette
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

# Output directory
OUTPUT_DIR = Path(__file__).parent

# ============================================================
# Diagram 1: Vectors as Arrows in 2D Space
# ============================================================
def create_vectors_geometric():
    """Geometric view of vectors as arrows from the origin."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define vectors
    v1 = np.array([3, 2])
    v2 = np.array([1, 4])
    v_sum = v1 + v2

    # Plot vectors as arrows from origin
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
              color=COLORS['blue'], width=0.008, label='v₁ = [3, 2]')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
              color=COLORS['red'], width=0.008, label='v₂ = [1, 4]')
    ax.quiver(0, 0, v_sum[0], v_sum[1], angles='xy', scale_units='xy', scale=1,
              color=COLORS['green'], width=0.008, label='v₁ + v₂ = [4, 6]')

    # Plot parallelogram showing vector addition
    ax.plot([v1[0], v_sum[0]], [v1[1], v_sum[1]], 'r--', alpha=0.3, linewidth=1)
    ax.plot([v2[0], v_sum[0]], [v2[1], v_sum[1]], 'b--', alpha=0.3, linewidth=1)

    # Formatting
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend(fontsize=10)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Geometric View: Vectors as Arrows in Space', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'vectors_geometric.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Created: {output_path.name}")

# ============================================================
# Diagram 2: The Feature Matrix X
# ============================================================
def create_feature_matrix():
    """Data science view of the feature matrix."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw matrix structure
    n_rows, n_cols = 5, 4
    cell_width, cell_height = 1, 0.8

    # Draw cells with example values
    sample_data = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [6.2, 2.9, 4.3, 1.3],
        [5.9, 3.0, 5.1, 1.8],
        [6.7, 3.1, 4.7, 1.5]
    ])

    for i in range(n_rows):
        for j in range(n_cols):
            rect = patches.Rectangle((j*cell_width, (n_rows-1-i)*cell_height),
                                     cell_width, cell_height,
                                     linewidth=1, edgecolor='black', facecolor='lightblue')
            ax.add_patch(rect)
            # Add values
            ax.text(j*cell_width + cell_width/2, (n_rows-1-i)*cell_height + cell_height/2,
                   f'{sample_data[i, j]:.1f}', ha='center', va='center', fontsize=10)

    # Highlight one row (sample)
    row_highlight = patches.Rectangle((0, 3*cell_height), n_cols*cell_width, cell_height,
                                     linewidth=3, edgecolor=COLORS['red'], facecolor='none')
    ax.add_patch(row_highlight)
    ax.text(n_cols*cell_width + 0.3, 3*cell_height + cell_height/2,
            '← One sample (row vector)', va='center', fontsize=11,
            color=COLORS['red'], fontweight='bold')

    # Highlight one column (feature)
    col_highlight = patches.Rectangle((0, 0), cell_width, n_rows*cell_height,
                                     linewidth=3, edgecolor=COLORS['blue'], facecolor='none')
    ax.add_patch(col_highlight)
    ax.text(cell_width/2, -0.5, 'One feature\n(column)', ha='center',
            fontsize=11, color=COLORS['blue'], fontweight='bold')

    # Add dimension labels
    ax.text(-0.8, n_rows*cell_height/2, f'n samples\n(rows)',
            ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    ax.text(n_cols*cell_width/2, n_rows*cell_height + 0.5, 'p features (columns)',
            ha='center', fontsize=12, fontweight='bold')

    # Add shape annotation
    ax.text(n_cols*cell_width + 0.3, n_rows*cell_height - 0.3,
            'X.shape = (n, p)', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Feature names
    feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
    for j, name in enumerate(feature_names):
        ax.text(j*cell_width + cell_width/2, n_rows*cell_height + 0.2,
               name, ha='center', fontsize=9, style='italic')

    ax.set_xlim(-1.5, 6)
    ax.set_ylim(-1, 5)
    ax.axis('off')
    ax.set_title('Data Science View: Every Dataset is a Feature Matrix X',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'feature_matrix.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Created: {output_path.name}")

# ============================================================
# Diagram 3: Matrix-Vector Multiplication Dimensions
# ============================================================
def create_matrix_vector_multiplication():
    """Visualization of matrix-vector multiplication dimensions."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw three matrices/vectors showing dimension matching
    # Matrix A (n × p)
    ax.add_patch(patches.Rectangle((0.5, 2), 1.5, 2,
                 linewidth=2, edgecolor=COLORS['blue'], facecolor='lightblue'))
    ax.text(1.25, 3, 'X', ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(1.25, 1.5, '(n × p)', ha='center', fontsize=12)
    ax.text(1.25, 5.2, 'n samples', ha='center', fontsize=10, color=COLORS['blue'])
    ax.text(2.3, 3, 'p features', ha='center', fontsize=10, color=COLORS['blue'], rotation=270)

    # Multiply symbol
    ax.text(2.8, 3, '×', ha='center', va='center', fontsize=24, fontweight='bold')

    # Vector v (p × 1)
    ax.add_patch(patches.Rectangle((3.5, 2), 0.5, 2,
                 linewidth=2, edgecolor=COLORS['red'], facecolor='lightcoral'))
    ax.text(3.75, 3, 'θ', ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(3.75, 1.5, '(p × 1)', ha='center', fontsize=12)
    ax.text(3.75, 5.2, 'p weights', ha='center', fontsize=10, color=COLORS['red'])

    # Equals symbol
    ax.text(4.7, 3, '=', ha='center', va='center', fontsize=24, fontweight='bold')

    # Result vector (n × 1)
    ax.add_patch(patches.Rectangle((5.5, 2), 0.5, 2,
                 linewidth=2, edgecolor=COLORS['green'], facecolor='lightgreen'))
    ax.text(5.75, 3, 'ŷ', ha='center', va='center', fontsize=20, fontweight='bold')
    ax.text(5.75, 1.5, '(n × 1)', ha='center', fontsize=12)
    ax.text(5.75, 5.2, 'n predictions', ha='center', fontsize=10, color=COLORS['green'])

    # Add dimension matching annotation
    ax.annotate('', xy=(3.5, 1.2), xytext=(2, 1.2),
                arrowprops=dict(arrowstyle='<->', color=COLORS['purple'], lw=2))
    ax.text(2.75, 0.9, 'Inner dimensions must match!', ha='center',
            fontsize=11, color=COLORS['purple'], fontweight='bold')

    # Add example with numbers
    ax.text(1.25, 0.3, 'Example:\n(150 × 4)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(3.75, 0.3, '(4 × 1)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    ax.text(5.75, 0.3, '(150 × 1)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Add interpretation
    ax.text(8.5, 3.5, 'Interpretation:', fontsize=12, fontweight='bold')
    ax.text(8.5, 2.8, '• X = data matrix (150 flowers, 4 features)', fontsize=10, ha='left')
    ax.text(8.5, 2.3, '• θ = weight vector (4 coefficients)', fontsize=10, ha='left')
    ax.text(8.5, 1.8, '• ŷ = predictions (150 outputs)', fontsize=10, ha='left')
    ax.text(8.5, 1.2, 'This is how model.predict() works!', fontsize=11, ha='left',
            style='italic', color='darkgreen', fontweight='bold')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('Matrix-Vector Multiplication: The Prediction Formula',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'matrix_vector_multiplication.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Created: {output_path.name}")

# ============================================================
# Diagram 4: Iris Vectors in 2D
# ============================================================
def create_iris_vectors_2d():
    """Visualization of Iris flowers as 2D vectors."""
    from sklearn.datasets import load_iris

    # Load data
    iris = load_iris()
    X = iris.data

    # Use only 2 features for visualization
    X_2d = X[:, :2]  # Sepal length and sepal width

    # Plot first 3 flowers as vectors
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = [COLORS['blue'], COLORS['red'], COLORS['green']]
    for i in range(3):
        vector = X_2d[i]
        ax.quiver(0, 0, vector[0], vector[1],
                 angles='xy', scale_units='xy', scale=1,
                 color=colors[i], width=0.006,
                 label=f'Flower {i}: [{vector[0]:.1f}, {vector[1]:.1f}]')
        ax.scatter(vector[0], vector[1], color=colors[i], s=100, zorder=5)

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend(fontsize=9)
    ax.set_xlabel('Sepal Length (cm)', fontsize=12)
    ax.set_ylabel('Sepal Width (cm)', fontsize=12)
    ax.set_title('Iris Flowers as 2D Vectors', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'iris_vectors_2d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Created: {output_path.name}")

# ============================================================
# Main execution
# ============================================================
if __name__ == '__main__':
    print("Generating diagrams for Chapter 1: Linear Algebra")
    print("=" * 60)

    create_vectors_geometric()
    create_feature_matrix()
    create_matrix_vector_multiplication()
    create_iris_vectors_2d()

    print("=" * 60)
    print("All diagrams generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
