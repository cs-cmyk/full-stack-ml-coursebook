#!/usr/bin/env python3
"""Generate smoking/genetics medical DAG."""

import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
# Scenario: Smoking → Lung Cancer ← Genetic Factor → Chronic Cough
G = nx.DiGraph()

# Add nodes
nodes = ['Smoking', 'Genetic_Factor', 'Lung_Cancer', 'Chronic_Cough']
G.add_nodes_from(nodes)

# Add edges (causal relationships)
edges = [
    ('Smoking', 'Lung_Cancer'),
    ('Genetic_Factor', 'Lung_Cancer'),
    ('Genetic_Factor', 'Chronic_Cough')
]
G.add_edges_from(edges)

# Visualize the DAG
pos = {
    'Smoking': (0, 1),
    'Genetic_Factor': (2, 1),
    'Lung_Cancer': (1, 0),
    'Chronic_Cough': (3, 0)
}

plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_color='#2196F3',
        node_size=3000, font_size=10, font_weight='bold',
        arrows=True, arrowsize=20, arrowstyle='->',
        edge_color='gray', width=2)
plt.title('Causal DAG: Smoking, Genetics, and Health Outcomes',
          fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-19/ch54/diagrams/smoking_dag.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated smoking_dag.png")
