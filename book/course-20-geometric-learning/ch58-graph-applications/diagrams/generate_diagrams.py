"""
Generate all diagrams for Chapter 58: Applications of Graph ML
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Color palette
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

# ============================================================================
# Diagram 1: Application Domain Overview
# ============================================================================
def create_domain_overview():
    """Create overview of graph ML application domains"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Graph ML Application Domains',
            ha='center', va='top', fontsize=18, fontweight='bold')

    # Define domains with positions
    domains = [
        {'name': 'Molecular\nProperty\nPrediction', 'pos': (1.5, 7),
         'color': COLORS['blue'], 'graph': 'Homogeneous', 'task': 'Graph-level'},
        {'name': 'Fraud\nDetection', 'pos': (5, 7),
         'color': COLORS['red'], 'graph': 'Heterogeneous', 'task': 'Node Classification'},
        {'name': 'Knowledge\nGraphs', 'pos': (8.5, 7),
         'color': COLORS['green'], 'graph': 'Multi-Relational', 'task': 'Link Prediction'},
        {'name': 'Recommendation\nSystems', 'pos': (3.25, 3.5),
         'color': COLORS['orange'], 'graph': 'Bipartite', 'task': 'Link Prediction'},
        {'name': 'Traffic\nForecasting', 'pos': (6.75, 3.5),
         'color': COLORS['purple'], 'graph': 'Temporal', 'task': 'Time Series'}
    ]

    for domain in domains:
        x, y = domain['pos']
        # Main box
        box = FancyBboxPatch((x-0.8, y-0.8), 1.6, 1.6,
                            boxstyle="round,pad=0.1",
                            edgecolor=domain['color'],
                            facecolor=domain['color'],
                            alpha=0.3, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, domain['name'], ha='center', va='center',
                fontsize=11, fontweight='bold', color=domain['color'])

        # Graph type
        ax.text(x, y-1.3, f"Graph: {domain['graph']}",
                ha='center', va='top', fontsize=9, style='italic')

        # Task type
        ax.text(x, y-1.7, f"Task: {domain['task']}",
                ha='center', va='top', fontsize=9, style='italic')

    # Add central message
    ax.text(5, 0.5,
            'Key: Match graph structure to domain relationships',
            ha='center', va='bottom', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('/home/chirag/ds-book/book/course-20/ch58/diagrams/domain_overview.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created domain_overview.png")


# ============================================================================
# Diagram 2: Molecular Property Prediction Workflow
# ============================================================================
def create_molecular_workflow():
    """Illustrate molecular property prediction workflow"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Molecule representation
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('1. Molecule as Graph', fontsize=14, fontweight='bold')

    # Simple molecule graph
    nodes = [(3, 7), (5, 8), (7, 7), (7, 5), (5, 4), (3, 5)]
    node_labels = ['C', 'O', 'C', 'C', 'N', 'C']

    for i, (pos, label) in enumerate(zip(nodes, node_labels)):
        color = COLORS['blue'] if label == 'C' else (COLORS['red'] if label == 'O' else COLORS['green'])
        circle = Circle(pos, 0.4, color=color, alpha=0.6, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], label, ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

    # Edges
    edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)]
    for i, j in edges:
        x = [nodes[i][0], nodes[j][0]]
        y = [nodes[i][1], nodes[j][1]]
        ax.plot(x, y, 'k-', linewidth=2, alpha=0.5)

    ax.text(5, 1.5, 'Atoms = Nodes\nBonds = Edges',
            ha='center', va='top', fontsize=11, style='italic')

    # Panel 2: Message passing
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('2. Message Passing', fontsize=14, fontweight='bold')

    # Central node with neighbors
    center = (5, 5)
    neighbors = [(3, 6), (7, 6), (3, 4), (7, 4)]

    # Draw neighbors
    for pos in neighbors:
        circle = Circle(pos, 0.3, color=COLORS['gray'], alpha=0.4, ec='black', linewidth=1.5)
        ax.add_patch(circle)

    # Draw center node (highlighted)
    circle = Circle(center, 0.5, color=COLORS['orange'], alpha=0.8, ec='black', linewidth=2.5)
    ax.add_patch(circle)

    # Arrows showing message passing
    for pos in neighbors:
        arrow = FancyArrowPatch(pos, center,
                               arrowstyle='->', mutation_scale=20,
                               color=COLORS['purple'], linewidth=2, alpha=0.7)
        ax.add_patch(arrow)

    ax.text(5, 2, 'Aggregate neighbor\ninformation',
            ha='center', va='top', fontsize=11, style='italic')

    # Panel 3: Prediction
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('3. Global Pooling → Prediction', fontsize=14, fontweight='bold')

    # Multiple nodes
    node_positions = [(2, 7), (4, 8), (6, 7), (8, 6), (2, 5), (4, 4), (6, 5), (8, 4)]
    for pos in node_positions:
        circle = Circle(pos, 0.25, color=COLORS['blue'], alpha=0.6, ec='black', linewidth=1)
        ax.add_patch(circle)

    # Pooling arrow
    arrow = FancyArrowPatch((5, 6.5), (5, 4.5),
                           arrowstyle='->', mutation_scale=30,
                           color=COLORS['green'], linewidth=3)
    ax.add_patch(arrow)
    ax.text(6.5, 5.5, 'Pool', fontsize=11, fontweight='bold', color=COLORS['green'])

    # Graph embedding box
    box = FancyBboxPatch((3.5, 2.5), 3, 1,
                        boxstyle="round,pad=0.1",
                        edgecolor=COLORS['purple'],
                        facecolor=COLORS['purple'],
                        alpha=0.3, linewidth=2)
    ax.add_patch(box)
    ax.text(5, 3, 'Graph Embedding', ha='center', va='center', fontsize=10, fontweight='bold')

    # Prediction
    arrow = FancyArrowPatch((5, 2.3), (5, 1.5),
                           arrowstyle='->', mutation_scale=20,
                           color='black', linewidth=2)
    ax.add_patch(arrow)
    ax.text(5, 0.8, 'Solubility: -2.18', ha='center', va='top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig('/home/chirag/ds-book/book/course-20/ch58/diagrams/molecular_workflow.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created molecular_workflow.png")


# ============================================================================
# Diagram 3: Fraud Detection Graph Structure
# ============================================================================
def create_fraud_detection_graph():
    """Illustrate heterogeneous fraud detection graph"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'Fraud Detection: Heterogeneous Graph',
            ha='center', va='top', fontsize=16, fontweight='bold')

    # Define node types and positions
    users = [(2, 7), (3.5, 6), (2, 5), (3.5, 4)]  # Fraud ring
    legitimate_users = [(7, 7), (8.5, 6)]
    merchants = [(5, 7), (5, 4.5)]
    devices = [(2, 2.5), (8, 2.5)]

    # Draw fraud ring (users connected in circle)
    for i, pos in enumerate(users):
        circle = Circle(pos, 0.3, color=COLORS['red'], alpha=0.7, ec='darkred', linewidth=2.5)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], 'U', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    # Fraud ring connections (circular)
    for i in range(len(users)):
        start = users[i]
        end = users[(i+1) % len(users)]
        arrow = FancyArrowPatch(start, end,
                               arrowstyle='->', mutation_scale=15,
                               color=COLORS['red'], linewidth=2.5, alpha=0.8,
                               connectionstyle="arc3,rad=0.3")
        ax.add_patch(arrow)

    # Legitimate users
    for pos in legitimate_users:
        circle = Circle(pos, 0.3, color=COLORS['green'], alpha=0.6, ec='darkgreen', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], 'U', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    # Merchants
    for i, pos in enumerate(merchants):
        square = FancyBboxPatch((pos[0]-0.3, pos[1]-0.3), 0.6, 0.6,
                               boxstyle="round,pad=0.05",
                               edgecolor=COLORS['blue'],
                               facecolor=COLORS['blue'],
                               alpha=0.6, linewidth=2)
        ax.add_patch(square)
        ax.text(pos[0], pos[1], 'M', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    # Devices
    for i, pos in enumerate(devices):
        diamond_points = np.array([
            [pos[0], pos[1]+0.35],
            [pos[0]+0.35, pos[1]],
            [pos[0], pos[1]-0.35],
            [pos[0]-0.35, pos[1]]
        ])
        color = COLORS['red'] if i == 0 else COLORS['green']
        polygon = plt.Polygon(diamond_points, color=color, alpha=0.6,
                            ec='black', linewidth=2)
        ax.add_patch(polygon)
        ax.text(pos[0], pos[1], 'D', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

    # Connections: fraud users -> merchant -> device
    for user_pos in users[:2]:
        ax.plot([user_pos[0], merchants[0][0]], [user_pos[1], merchants[0][1]],
                'k--', linewidth=1.5, alpha=0.5)
    ax.plot([merchants[0][0], devices[0][0]], [merchants[0][1], devices[0][1]],
            color=COLORS['red'], linewidth=2, alpha=0.7, linestyle='--')

    # Legitimate connections
    for user_pos in legitimate_users:
        ax.plot([user_pos[0], merchants[1][0]], [user_pos[1], merchants[1][1]],
                'k--', linewidth=1.5, alpha=0.5)
    ax.plot([merchants[1][0], devices[1][0]], [merchants[1][1], devices[1][1]],
            color=COLORS['green'], linewidth=2, alpha=0.7, linestyle='--')

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['red'], label='Fraudulent Users (Ring)', alpha=0.7),
        mpatches.Patch(color=COLORS['green'], label='Legitimate Users', alpha=0.6),
        mpatches.Patch(color=COLORS['blue'], label='Merchants', alpha=0.6),
        mpatches.Patch(color=COLORS['gray'], label='Shared Device (Signal)', alpha=0.6)
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)

    # Annotation
    ax.text(2.75, 8.5, 'Fraud Ring:\nCircular transfers\n+ shared device',
            ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor=COLORS['red'], alpha=0.3))

    plt.tight_layout()
    plt.savefig('/home/chirag/ds-book/book/course-20/ch58/diagrams/fraud_detection_graph.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created fraud_detection_graph.png")


# ============================================================================
# Diagram 4: Knowledge Graph Embeddings (TransE)
# ============================================================================
def create_knowledge_graph_embeddings():
    """Visualize TransE embedding space"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Triple structure
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Knowledge Graph Triples', fontsize=14, fontweight='bold')

    # Example triples
    triples = [
        ('Paris', 'capital_of', 'France', 2),
        ('Berlin', 'capital_of', 'Germany', 5),
        ('France', 'neighbor_of', 'Germany', 8)
    ]

    for head, rel, tail, y_pos in triples:
        # Head
        circle = Circle((1.5, y_pos), 0.4, color=COLORS['blue'], alpha=0.7, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(1.5, y_pos, head[:3], ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

        # Relation arrow
        arrow = FancyArrowPatch((2.2, y_pos), (4.8, y_pos),
                               arrowstyle='->', mutation_scale=20,
                               color=COLORS['orange'], linewidth=2.5)
        ax.add_patch(arrow)
        ax.text(3.5, y_pos+0.4, rel, ha='center', va='bottom',
                fontsize=9, style='italic', color=COLORS['orange'])

        # Tail
        circle = Circle((5.5, y_pos), 0.4, color=COLORS['green'], alpha=0.7, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(5.5, y_pos, tail[:3], ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

    ax.text(3.5, 0.5, 'Goal: Learn embeddings where h + r ≈ t',
            ha='center', va='bottom', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 2: TransE embedding space
    ax = axes[1]
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.set_xlabel('Embedding Dimension 1', fontsize=12)
    ax.set_ylabel('Embedding Dimension 2', fontsize=12)
    ax.set_title('TransE Embedding Space', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Entities
    entities = {
        'Paris': np.array([0, 0]),
        'France': np.array([1, 0.2]),
        'Berlin': np.array([0.2, 1.5]),
        'Germany': np.array([1.2, 1.7])
    }

    # Relation vector
    relation_vec = np.array([1, 0.2])

    # Plot entities
    for name, pos in entities.items():
        if 'Paris' in name or 'Berlin' in name:
            color = COLORS['blue']
            marker = 'o'
        else:
            color = COLORS['green']
            marker = 's'
        ax.scatter(pos[0], pos[1], s=200, color=color, alpha=0.7,
                  edgecolors='black', linewidth=2, marker=marker, zorder=3)
        ax.text(pos[0], pos[1]-0.25, name, ha='center', va='top', fontsize=10, fontweight='bold')

    # Show translation: Paris + capital_of ≈ France
    ax.arrow(0, 0, 0.9, 0.18, head_width=0.15, head_length=0.1,
            fc=COLORS['orange'], ec='black', linewidth=2, alpha=0.8, zorder=2)
    ax.text(0.5, 0.4, 'capital_of', fontsize=10, color=COLORS['orange'],
            fontweight='bold', style='italic')

    # Show translation: Berlin + capital_of ≈ Germany
    ax.arrow(0.2, 1.5, 0.9, 0.18, head_width=0.15, head_length=0.1,
            fc=COLORS['orange'], ec='black', linewidth=2, alpha=0.8, zorder=2)

    plt.tight_layout()
    plt.savefig('/home/chirag/ds-book/book/course-20/ch58/diagrams/knowledge_graph_embeddings.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created knowledge_graph_embeddings.png")


# ============================================================================
# Diagram 5: Recommendation System Bipartite Graph
# ============================================================================
def create_recommendation_bipartite():
    """Illustrate bipartite user-item graph for recommendations"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'Recommendation: Bipartite User-Item Graph',
            ha='center', va='top', fontsize=16, fontweight='bold')

    # Users on left
    users = [(2, 7), (2, 5.5), (2, 4), (2, 2.5)]
    user_labels = ['Alice', 'Bob', 'Carol', 'Dave']

    for pos, label in zip(users, user_labels):
        circle = Circle(pos, 0.35, color=COLORS['blue'], alpha=0.7, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos[0]-0.9, pos[1], label, ha='right', va='center',
                fontsize=11, fontweight='bold')

    # Items on right
    items = [(8, 7.5), (8, 6), (8, 4.5), (8, 3)]
    item_labels = ['Movie A', 'Movie B', 'Movie C', 'Movie D']

    for pos, label in zip(items, item_labels):
        square = FancyBboxPatch((pos[0]-0.35, pos[1]-0.35), 0.7, 0.7,
                               boxstyle="round,pad=0.05",
                               edgecolor=COLORS['orange'],
                               facecolor=COLORS['orange'],
                               alpha=0.7, linewidth=2)
        ax.add_patch(square)
        ax.text(pos[0]+0.9, pos[1], label, ha='left', va='center',
                fontsize=11, fontweight='bold')

    # Interactions (edges)
    interactions = [
        (0, 0, 5), (0, 1, 4),  # Alice
        (1, 1, 5), (1, 2, 3),  # Bob
        (2, 0, 4), (2, 3, 5),  # Carol
        (3, 2, 4), (3, 3, 4)   # Dave
    ]

    for user_idx, item_idx, rating in interactions:
        user_pos = users[user_idx]
        item_pos = items[item_idx]
        alpha = rating / 5.0
        ax.plot([user_pos[0], item_pos[0]], [user_pos[1], item_pos[1]],
                color=COLORS['purple'], linewidth=2*alpha, alpha=alpha)

    # Prediction (dashed line)
    ax.plot([users[1][0], items[0][0]], [users[1][1], items[0][1]],
            color=COLORS['green'], linewidth=2.5, linestyle='--', alpha=0.8)
    ax.text(5, 6.7, 'Predict', fontsize=10, color=COLORS['green'],
            fontweight='bold', ha='center', va='bottom')

    # Message passing illustration
    ax.annotate('', xy=(5, 5), xytext=(2.5, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2, alpha=0.6))
    ax.annotate('', xy=(5, 5), xytext=(7.5, 6),
                arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2, alpha=0.6))
    ax.text(5, 5, 'Message\nPassing', ha='center', va='center',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

    # Annotation
    ax.text(5, 1, 'GNN propagates preferences through graph\nCold-start solved via collaborative filtering',
            ha='center', va='top', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('/home/chirag/ds-book/book/course-20/ch58/diagrams/recommendation_bipartite.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created recommendation_bipartite.png")


# ============================================================================
# Diagram 6: Performance Comparison Across Domains
# ============================================================================
def create_performance_comparison():
    """Compare GNN vs baseline performance across domains"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Data for comparisons
    domains = ['Molecular\nPrediction', 'Fraud\nDetection', 'Knowledge\nGraphs', 'Recommendations']
    gnn_scores = [0.712, 0.842, 0.867, 0.418]
    baseline_scores = [0.783, 0.444, 0.650, 0.356]
    metrics = ['RMSE\n(lower better)', 'F1 Score', 'MRR', 'NDCG@10']

    # For RMSE, lower is better, so we'll show improvement differently
    improvements = []
    for i, (gnn, base) in enumerate(zip(gnn_scores, baseline_scores)):
        if i == 0:  # RMSE - lower is better
            improv = ((base - gnn) / base) * 100
        else:
            improv = ((gnn - base) / base) * 100
        improvements.append(improv)

    # Plot 1: Bar comparison
    ax = axes[0, 0]
    x = np.arange(len(domains))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline',
                   color=COLORS['gray'], alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, gnn_scores, width, label='GNN',
                   color=COLORS['blue'], alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('GNN vs Baseline Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add metric labels
    for i, metric in enumerate(metrics):
        ax.text(i, -0.15, metric, ha='center', va='top', fontsize=8, style='italic')

    # Plot 2: Improvement percentages
    ax = axes[0, 1]
    colors_bars = [COLORS['green'] if imp > 0 else COLORS['red'] for imp in improvements]
    bars = ax.barh(domains, improvements, color=colors_bars, alpha=0.7,
                   edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('GNN Improvement Over Baseline', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=2)
    ax.grid(axis='x', alpha=0.3)

    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax.text(val + 2, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

    # Plot 3: When does graph structure help?
    ax = axes[1, 0]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.text(5, 9, 'When Does Graph Structure Help?',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # Positive cases
    y_pos = 7.5
    ax.text(0.5, y_pos, '✓ Helps:', ha='left', va='top',
            fontsize=12, fontweight='bold', color=COLORS['green'])

    helps_cases = [
        'Relationships contain signal beyond features',
        'Multi-hop patterns matter (fraud rings)',
        'Transitive reasoning (knowledge graphs)',
        'Cold-start problems (recommendations)'
    ]
    for i, case in enumerate(helps_cases):
        ax.text(1, y_pos - 0.7*(i+1), f'• {case}', ha='left', va='top', fontsize=10)

    # Negative cases
    y_pos = 3.5
    ax.text(0.5, y_pos, '✗ Doesn\'t Help:', ha='left', va='top',
            fontsize=12, fontweight='bold', color=COLORS['red'])

    no_help_cases = [
        'Features are sufficient (no relational signal)',
        'Graph is too sparse or noisy',
        'Independent instances (like coin flips)'
    ]
    for i, case in enumerate(no_help_cases):
        ax.text(1, y_pos - 0.7*(i+1), f'• {case}', ha='left', va='top', fontsize=10)

    # Plot 4: Scalability considerations
    ax = axes[1, 1]

    graph_sizes = ['Small\n(<10K nodes)', 'Medium\n(10K-100K)', 'Large\n(100K-1M)', 'Very Large\n(>1M)']
    methods = ['Full-batch GNN', 'Neighbor Sampling', 'Cluster Sampling', 'Pre-computed']

    # Suitability matrix (0-1 scale)
    suitability = np.array([
        [1.0, 0.8, 0.3, 0.1],  # Full-batch
        [0.6, 1.0, 0.9, 0.6],  # Neighbor sampling
        [0.4, 0.8, 1.0, 0.8],  # Cluster sampling
        [0.7, 0.9, 0.9, 1.0]   # Pre-computed
    ])

    im = ax.imshow(suitability, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(graph_sizes)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(graph_sizes, fontsize=10)
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_title('Scalability: Method Selection', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(graph_sizes)):
            text = ax.text(j, i, f'{suitability[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Suitability')

    plt.tight_layout()
    plt.savefig('/home/chirag/ds-book/book/course-20/ch58/diagrams/performance_comparison.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created performance_comparison.png")


# ============================================================================
# Diagram 7: Application Selection Decision Tree
# ============================================================================
def create_decision_tree():
    """Create decision tree for choosing graph ML approach"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(7, 9.5, 'Graph ML Application Selection Guide',
            ha='center', va='top', fontsize=16, fontweight='bold')

    # Root question
    box = FancyBboxPatch((5, 8.2), 4, 0.8,
                        boxstyle="round,pad=0.1",
                        edgecolor='black',
                        facecolor=COLORS['blue'],
                        alpha=0.3, linewidth=2)
    ax.add_patch(box)
    ax.text(7, 8.6, 'Do relationships exist\nbetween data points?',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Branch: No
    ax.plot([5.5, 3], [8.2, 7.2], 'k-', linewidth=2)
    ax.text(4, 7.8, 'No', fontsize=10, fontweight='bold')

    box = FancyBboxPatch((1.5, 6.5), 3, 0.6,
                        boxstyle="round,pad=0.1",
                        edgecolor='black',
                        facecolor=COLORS['red'],
                        alpha=0.3, linewidth=2)
    ax.add_patch(box)
    ax.text(3, 6.8, 'Use tabular ML\n(RF, XGBoost, NN)',
            ha='center', va='center', fontsize=10)

    # Branch: Yes
    ax.plot([8.5, 11], [8.2, 7.2], 'k-', linewidth=2)
    ax.text(10, 7.8, 'Yes', fontsize=10, fontweight='bold')

    # Second level: Graph type
    box = FancyBboxPatch((9, 6.5), 4, 0.6,
                        boxstyle="round,pad=0.1",
                        edgecolor='black',
                        facecolor=COLORS['orange'],
                        alpha=0.3, linewidth=2)
    ax.add_patch(box)
    ax.text(11, 6.8, 'What is your task?',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Task branches
    tasks = [
        ('Node\nClassification', 3, 5, 'GCN, GAT\n(Semi-supervised)'),
        ('Graph\nClassification', 7, 5, 'GNN + Pooling\n(Molecules)'),
        ('Link\nPrediction', 11, 5, 'TransE, LightGCN\n(KG, RecSys)')
    ]

    for task_name, x_pos, y_pos, method in tasks:
        # Arrow from parent
        ax.plot([11, x_pos+1], [6.5, y_pos+0.5], 'k-', linewidth=1.5)

        # Task box
        box = FancyBboxPatch((x_pos, y_pos), 2, 0.5,
                            boxstyle="round,pad=0.05",
                            edgecolor='black',
                            facecolor=COLORS['green'],
                            alpha=0.3, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x_pos+1, y_pos+0.25, task_name,
                ha='center', va='center', fontsize=9, fontweight='bold')

        # Method box
        ax.plot([x_pos+1, x_pos+1], [y_pos, y_pos-0.7], 'k-', linewidth=1.5)
        box = FancyBboxPatch((x_pos-0.2, y_pos-1.5), 2.4, 0.6,
                            boxstyle="round,pad=0.05",
                            edgecolor=COLORS['purple'],
                            facecolor=COLORS['purple'],
                            alpha=0.2, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x_pos+1, y_pos-1.2, method,
                ha='center', va='center', fontsize=8, style='italic')

    # Bottom considerations
    ax.text(7, 2.5, 'Key Considerations:',
            ha='center', va='top', fontsize=12, fontweight='bold')

    considerations = [
        '1. Always compare against non-graph baselines',
        '2. Match graph structure to domain relationships',
        '3. Consider scalability (sampling for large graphs)',
        '4. Use appropriate metrics (F1 for imbalance, NDCG for ranking)',
        '5. Validate that graph structure provides signal'
    ]

    for i, text in enumerate(considerations):
        ax.text(1, 2.0 - i*0.35, text, ha='left', va='top', fontsize=9)

    plt.tight_layout()
    plt.savefig('/home/chirag/ds-book/book/course-20/ch58/diagrams/decision_tree.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created decision_tree.png")


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    print("Generating diagrams for Chapter 58: Applications of Graph ML\n")

    create_domain_overview()
    create_molecular_workflow()
    create_fraud_detection_graph()
    create_knowledge_graph_embeddings()
    create_recommendation_bipartite()
    create_performance_comparison()
    create_decision_tree()

    print("\n✓ All diagrams generated successfully!")
