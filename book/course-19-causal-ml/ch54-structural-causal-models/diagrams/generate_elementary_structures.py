#!/usr/bin/env python3
"""Generate elementary structures diagram for DAGs chapter."""

import matplotlib.pyplot as plt
import networkx as nx

# Create figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Color palette
BLUE = '#2196F3'
GREEN = '#4CAF50'
RED = '#F44336'

# 1. Chain: X → Y → Z
G_chain = nx.DiGraph()
G_chain.add_edges_from([('X', 'Y'), ('Y', 'Z')])
pos_chain = {'X': (0, 0), 'Y': (1, 0), 'Z': (2, 0)}

axes[0].set_title('Chain: Mediation\nX → Y → Z\n\nX ⊥ Z | Y (conditioning blocks)',
                  fontsize=12, fontweight='bold')
nx.draw(G_chain, pos_chain, ax=axes[0], with_labels=True,
        node_color=BLUE, node_size=2000, font_size=14,
        font_weight='bold', arrows=True, arrowsize=20,
        arrowstyle='->', edge_color='black', width=2)
axes[0].axis('off')

# 2. Fork: X ← Y → Z
G_fork = nx.DiGraph()
G_fork.add_edges_from([('Y', 'X'), ('Y', 'Z')])
pos_fork = {'X': (0, 0), 'Y': (1, 0.5), 'Z': (2, 0)}

axes[1].set_title('Fork: Confounding\nX ← Y → Z\n\nX ⊥ Z | Y (conditioning blocks)',
                  fontsize=12, fontweight='bold')
nx.draw(G_fork, pos_fork, ax=axes[1], with_labels=True,
        node_color=GREEN, node_size=2000, font_size=14,
        font_weight='bold', arrows=True, arrowsize=20,
        arrowstyle='->', edge_color='black', width=2)
axes[1].axis('off')

# 3. Collider: X → Y ← Z
G_collider = nx.DiGraph()
G_collider.add_edges_from([('X', 'Y'), ('Z', 'Y')])
pos_collider = {'X': (0, 0), 'Y': (1, -0.5), 'Z': (2, 0)}

axes[2].set_title('Collider: Selection Bias\nX → Y ← Z\n\nX ⊥̸ Z | Y (conditioning opens!)',
                  fontsize=12, fontweight='bold', color=RED)
nx.draw(G_collider, pos_collider, ax=axes[2], with_labels=True,
        node_color=RED, node_size=2000, font_size=14,
        font_weight='bold', arrows=True, arrowsize=20,
        arrowstyle='->', edge_color='black', width=2)
axes[2].axis('off')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-19/ch54/diagrams/elementary_structures.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated elementary_structures.png")
