import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Create a simple graph for visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Create a small graph
G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)])
pos = nx.spring_layout(G, seed=42)

# Initial features (Layer 0)
initial_features = {0: 1.0, 1: 0.5, 2: 0.8, 3: 0.3, 4: 0.9}

# Layer 1: After first aggregation
layer1_features = {}
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    layer1_features[node] = np.mean([initial_features[n] for n in neighbors + [node]])

# Layer 2: After second aggregation
layer2_features = {}
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    layer2_features[node] = np.mean([layer1_features[n] for n in neighbors + [node]])

# Plot each layer
for idx, (features, title) in enumerate([
    (initial_features, 'Layer 0: Initial Features'),
    (layer1_features, 'Layer 1: After Aggregation'),
    (layer2_features, 'Layer 2: Information Spread')
]):
    ax = axes[idx]
    node_colors = [features[n] for n in G.nodes()]
    nx.draw(G, pos, node_color=node_colors, node_size=800, cmap='viridis',
            with_labels=True, ax=ax, vmin=0, vmax=1)
    ax.set_title(title, fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-20/ch57/diagrams/message_passing_visualization.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Message passing visualization saved")
