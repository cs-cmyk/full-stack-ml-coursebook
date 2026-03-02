"""
Generate Graph Terminology Visual Dictionary
Shows fundamental graph concepts with labeled examples
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
import numpy as np

# Color palette
BLUE = '#2196F3'
GREEN = '#4CAF50'
ORANGE = '#FF9800'
RED = '#F44336'
PURPLE = '#9C27B0'
GRAY = '#607D8B'

fig = plt.figure(figsize=(16, 10))

# ============= Panel 1: Basic Terminology (Top Left) =============
ax1 = plt.subplot(2, 3, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Basic Components', fontsize=14, fontweight='bold', pad=10)

# Node/Vertex
node_A = Circle((2, 7), 0.4, color=BLUE, ec='black', linewidth=2, zorder=3)
ax1.add_patch(node_A)
ax1.text(2, 7, 'A', ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=4)
ax1.text(2, 5.5, 'Node/Vertex', ha='center', fontsize=11, fontweight='bold')
ax1.text(2, 5, '(entity)', ha='center', fontsize=9, style='italic', color=GRAY)

# Edge/Link
node_B = Circle((7, 7), 0.4, color=BLUE, ec='black', linewidth=2, zorder=3)
ax1.add_patch(node_B)
ax1.text(7, 7, 'B', ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=4)
ax1.plot([2.4, 6.6], [7, 7], 'k-', linewidth=3, zorder=2)
ax1.text(4.5, 7.5, 'Edge/Link', ha='center', fontsize=11, fontweight='bold')
ax1.text(4.5, 5.5, 'Edge/Link', ha='center', fontsize=11, fontweight='bold')
ax1.text(4.5, 5, '(relationship)', ha='center', fontsize=9, style='italic', color=GRAY)

# Directed edge
node_C = Circle((2, 2.5), 0.4, color=BLUE, ec='black', linewidth=2, zorder=3)
ax1.add_patch(node_C)
ax1.text(2, 2.5, 'C', ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=4)

node_D = Circle((7, 2.5), 0.4, color=BLUE, ec='black', linewidth=2, zorder=3)
ax1.add_patch(node_D)
ax1.text(7, 2.5, 'D', ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=4)

arrow = FancyArrowPatch((2.4, 2.5), (6.6, 2.5), arrowstyle='->',
                        mutation_scale=30, linewidth=3, color=RED, zorder=2)
ax1.add_patch(arrow)
ax1.text(4.5, 3, 'Directed Edge', ha='center', fontsize=11, fontweight='bold')
ax1.text(4.5, 1, '(one-way)', ha='center', fontsize=9, style='italic', color=GRAY)

# ============= Panel 2: Advanced Components (Top Middle) =============
ax2 = plt.subplot(2, 3, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Special Features', fontsize=14, fontweight='bold', pad=10)

# Weighted edge
node_E = Circle((2, 7), 0.4, color=BLUE, ec='black', linewidth=2, zorder=3)
ax2.add_patch(node_E)
ax2.text(2, 7, 'E', ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=4)

node_F = Circle((7, 7), 0.4, color=BLUE, ec='black', linewidth=2, zorder=3)
ax2.add_patch(node_F)
ax2.text(7, 7, 'F', ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=4)

ax2.plot([2.4, 6.6], [7, 7], 'k-', linewidth=3, zorder=2)
# Weight label on edge
weight_box = FancyBboxPatch((4.2, 6.7), 0.8, 0.6, boxstyle="round,pad=0.05",
                            facecolor='yellow', edgecolor='black', linewidth=2, zorder=4)
ax2.add_patch(weight_box)
ax2.text(4.6, 7, '5', ha='center', va='center', fontsize=12, fontweight='bold', zorder=5)
ax2.text(4.5, 5.5, 'Weighted Edge', ha='center', fontsize=11, fontweight='bold')
ax2.text(4.5, 5, '(distance, cost, etc.)', ha='center', fontsize=9, style='italic', color=GRAY)

# Self-loop
node_G = Circle((4.5, 2.5), 0.4, color=BLUE, ec='black', linewidth=2, zorder=3)
ax2.add_patch(node_G)
ax2.text(4.5, 2.5, 'G', ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=4)

# Draw self-loop as arc
theta = np.linspace(0, 2*np.pi*0.7, 50)
loop_r = 0.8
loop_x = 4.5 + loop_r * np.cos(theta + np.pi/4)
loop_y = 2.5 + loop_r * np.sin(theta + np.pi/4)
ax2.plot(loop_x, loop_y, 'k-', linewidth=3, zorder=2)
# Add arrowhead
arrow_loop = FancyArrowPatch((loop_x[-5], loop_y[-5]), (loop_x[-1], loop_y[-1]),
                             arrowstyle='->', mutation_scale=20, linewidth=3, color='black', zorder=2)
ax2.add_patch(arrow_loop)

ax2.text(4.5, 0.5, 'Self-Loop', ha='center', fontsize=11, fontweight='bold')
ax2.text(4.5, 0, '(node to itself)', ha='center', fontsize=9, style='italic', color=GRAY)

# ============= Panel 3: Path and Cycle (Top Right) =============
ax3 = plt.subplot(2, 3, 3)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('Paths and Cycles', fontsize=14, fontweight='bold', pad=10)

# Path example
path_nodes = {
    'A': (1, 7.5),
    'B': (3, 7.5),
    'C': (5, 7.5),
    'D': (7, 7.5)
}

for label, (x, y) in path_nodes.items():
    if label in ['A', 'B', 'C', 'D']:
        color = GREEN
    else:
        color = BLUE
    node = Circle((x, y), 0.35, color=color, ec='black', linewidth=2, zorder=3)
    ax3.add_patch(node)
    ax3.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', color='white', zorder=4)

# Highlight path
ax3.plot([1.35, 2.65], [7.5, 7.5], color=GREEN, linewidth=4, zorder=2)
ax3.plot([3.35, 4.65], [7.5, 7.5], color=GREEN, linewidth=4, zorder=2)
ax3.plot([5.35, 6.65], [7.5, 7.5], color=GREEN, linewidth=4, zorder=2)

ax3.text(4, 6, 'Path: A→B→C→D', ha='center', fontsize=11, fontweight='bold')
ax3.text(4, 5.5, '(sequence of edges)', ha='center', fontsize=9, style='italic', color=GRAY)

# Cycle example
cycle_nodes = {
    'E': (2, 2.5),
    'F': (4.5, 3.5),
    'G': (4.5, 1.5)
}

for label, (x, y) in cycle_nodes.items():
    node = Circle((x, y), 0.35, color=ORANGE, ec='black', linewidth=2, zorder=3)
    ax3.add_patch(node)
    ax3.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', color='white', zorder=4)

# Cycle edges
ax3.plot([2.3, 4.2], [2.65, 3.35], color=ORANGE, linewidth=4, zorder=2)
ax3.plot([4.5, 4.5], [3.15, 1.85], color=ORANGE, linewidth=4, zorder=2)
ax3.plot([4.2, 2.35], [1.5, 2.35], color=ORANGE, linewidth=4, zorder=2)

ax3.text(3.5, 0.5, 'Cycle: E→F→G→E', ha='center', fontsize=11, fontweight='bold')
ax3.text(3.5, 0, '(returns to start)', ha='center', fontsize=9, style='italic', color=GRAY)

# ============= Panel 4: Graph Visualization (Bottom Left) =============
ax4 = plt.subplot(2, 3, 4)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('Example Graph', fontsize=14, fontweight='bold', pad=10)

# 6-node graph
graph_nodes = {
    0: (2, 7),
    1: (5, 8),
    2: (8, 7),
    3: (2, 4),
    4: (5, 3),
    5: (8, 4)
}

edges = [(0, 1), (0, 3), (1, 2), (1, 3), (1, 4), (2, 5), (3, 4), (4, 5)]

# Draw edges first
for u, v in edges:
    x1, y1 = graph_nodes[u]
    x2, y2 = graph_nodes[v]
    ax4.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=1, alpha=0.6)

# Draw nodes
for node, (x, y) in graph_nodes.items():
    circle = Circle((x, y), 0.4, color=BLUE, ec='black', linewidth=2, zorder=3)
    ax4.add_patch(circle)
    ax4.text(x, y, str(node), ha='center', va='center', fontsize=12,
             fontweight='bold', color='white', zorder=4)

# Add clique highlight
clique_nodes = [1, 3, 4]
for i, n1 in enumerate(clique_nodes):
    for n2 in clique_nodes[i+1:]:
        if (n1, n2) in edges or (n2, n1) in edges:
            x1, y1 = graph_nodes[n1]
            x2, y2 = graph_nodes[n2]
            ax4.plot([x1, x2], [y1, y2], color=RED, linewidth=3, zorder=2, alpha=0.7)

ax4.text(5, 1, 'Clique (nodes 1,3,4)', ha='center', fontsize=11,
         fontweight='bold', color=RED)
ax4.text(5, 0.5, 'all nodes fully connected', ha='center', fontsize=9,
         style='italic', color=GRAY)

# ============= Panel 5: Adjacency Matrix (Bottom Middle) =============
ax5 = plt.subplot(2, 3, 5)
ax5.set_xlim(-0.5, 6.5)
ax5.set_ylim(-0.5, 6.5)
ax5.axis('off')
ax5.set_title('Adjacency Matrix', fontsize=14, fontweight='bold', pad=10)

# Create adjacency matrix
adj_matrix = np.zeros((6, 6))
for u, v in edges:
    adj_matrix[u][v] = 1
    adj_matrix[v][u] = 1  # Undirected

# Draw matrix
cell_size = 0.9
for i in range(6):
    for j in range(6):
        color = 'lightblue' if adj_matrix[i][j] == 1 else 'white'
        rect = mpatches.Rectangle((j, 5-i), cell_size, cell_size,
                                  facecolor=color, edgecolor='black', linewidth=1.5)
        ax5.add_patch(rect)
        ax5.text(j + cell_size/2, 5-i + cell_size/2, str(int(adj_matrix[i][j])),
                ha='center', va='center', fontsize=12, fontweight='bold')

# Row and column labels
for i in range(6):
    ax5.text(-0.3, 5-i + cell_size/2, str(i), ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax5.text(i + cell_size/2, 6.3, str(i), ha='center', va='center',
            fontsize=12, fontweight='bold')

ax5.text(-0.3, 7, 'to', ha='center', va='center', fontsize=10, style='italic')
ax5.text(3, -0.8, 'from', ha='center', va='center', fontsize=10, style='italic')

# ============= Panel 6: Adjacency List (Bottom Right) =============
ax6 = plt.subplot(2, 3, 6)
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 10)
ax6.axis('off')
ax6.set_title('Adjacency List', fontsize=14, fontweight='bold', pad=10)

# Build adjacency list
adj_list = {i: [] for i in range(6)}
for u, v in edges:
    adj_list[u].append(v)
    adj_list[v].append(u)

# Sort for consistency
for node in adj_list:
    adj_list[node].sort()

# Display adjacency list
y_start = 8.5
line_height = 1.2
for node, neighbors in adj_list.items():
    y = y_start - node * line_height
    neighbors_str = ', '.join(map(str, neighbors))

    # Node box
    node_box = FancyBboxPatch((0.5, y-0.3), 0.8, 0.6, boxstyle="round,pad=0.05",
                              facecolor=BLUE, edgecolor='black', linewidth=2)
    ax6.add_patch(node_box)
    ax6.text(0.9, y, str(node), ha='center', va='center', fontsize=12,
            fontweight='bold', color='white')

    # Arrow
    ax6.text(1.8, y, '→', ha='center', va='center', fontsize=16, fontweight='bold')

    # Neighbors
    ax6.text(2.5, y, f'[{neighbors_str}]', ha='left', va='center', fontsize=11,
            family='monospace')

ax6.text(5, 0.8, 'Each node lists its neighbors', ha='center', fontsize=10,
        style='italic', color=GRAY)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-10-advanced/ch38-graphs/diagrams/graph_terminology.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Graph terminology diagram saved successfully!")
