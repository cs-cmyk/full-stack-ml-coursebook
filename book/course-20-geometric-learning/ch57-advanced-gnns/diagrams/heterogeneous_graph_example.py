import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Create heterogeneous graph visualization
G = nx.Graph()

# Add nodes
num_movies = 10
num_actors = 15
num_directors = 5

for i in range(num_movies):
    G.add_node(f'M{i}', node_type='movie')
for i in range(num_actors):
    G.add_node(f'A{i}', node_type='actor')
for i in range(num_directors):
    G.add_node(f'D{i}', node_type='director')

# Add edges (simplified for clarity)
np.random.seed(42)
# Stars-in edges
for i in range(20):
    actor = f'A{np.random.randint(0, num_actors)}'
    movie = f'M{np.random.randint(0, num_movies)}'
    G.add_edge(actor, movie, edge_type='stars_in')

# Directs edges
for i in range(num_directors):
    movie = f'M{np.random.randint(0, num_movies)}'
    G.add_edge(f'D{i}', movie, edge_type='directs')

# Create visualization
pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)
fig, ax = plt.subplots(figsize=(14, 10))

# Draw nodes by type
movie_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'movie']
actor_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'actor']
director_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'director']

nx.draw_networkx_nodes(G, pos, nodelist=movie_nodes, node_color='#2196F3',
                       node_size=800, label='Movies', ax=ax, alpha=0.8)
nx.draw_networkx_nodes(G, pos, nodelist=actor_nodes, node_color='#4CAF50',
                       node_size=600, label='Actors', ax=ax, alpha=0.8)
nx.draw_networkx_nodes(G, pos, nodelist=director_nodes, node_color='#FF9800',
                       node_size=600, label='Directors', ax=ax, alpha=0.8)

# Draw edges by type
stars_in_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'stars_in']
directs_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'directs']

nx.draw_networkx_edges(G, pos, edgelist=stars_in_edges, edge_color='#4CAF50',
                       width=2, alpha=0.5, label='stars_in', ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=directs_edges, edge_color='#FF9800',
                       width=2, alpha=0.5, style='dashed', label='directs', ax=ax)

nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
ax.set_title('Heterogeneous Movie Graph\n(Movies, Actors, Directors with Different Edge Types)',
            fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-20/ch57/diagrams/heterogeneous_graph_example.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Heterogeneous graph visualization saved")
