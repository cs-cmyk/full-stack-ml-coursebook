> **© 2026 Chirag Shinde. Licensed under CC BY-NC-SA 4.0.**
> See [LICENSE](../../LICENSE) for details.

---

# 38.1: Graph Theory Fundamentals

## Why This Matters

Social networks connect billions of people, molecules bond atoms in specific patterns, citation networks link scientific papers, and transportation systems connect cities. All of these systems share a common structure: entities connected by relationships. Graph theory provides the mathematical framework for analyzing these networks, enabling everything from friend recommendations on social media to drug discovery in pharmaceutical research. Understanding graphs unlocks the ability to analyze structure and relationships in data that traditional tabular methods cannot capture.

## Intuition

Think of a graph as a map of friendships at a party. Each person is a node (or vertex), and each friendship is an edge (or link) connecting two people. If Alice and Bob are friends, there's an edge between their nodes. The number of friends someone has is their degree. Finding out if two people know each other through mutual friends is like finding a path in the graph.

This same structure applies everywhere: web pages (nodes) connected by hyperlinks (edges), cities (nodes) connected by roads (edges), or proteins (nodes) connected by interactions (edges). Unlike a spreadsheet where each row is independent, graphs explicitly model relationships. The power of graph theory is that a single mathematical framework can represent and analyze all these different systems using the same concepts and algorithms.

In a social network, some people are more "important" than others—maybe they have many friends (high degree), or they bridge different groups (high betweenness), or they're friends with other important people (high eigenvector centrality). Graph theory gives us precise ways to measure these intuitive notions. Similarly, communities or clusters in a graph represent groups of nodes that are densely connected internally but sparsely connected externally—like departments in a company or friend groups at school.

The key insight: many real-world problems are fundamentally about understanding structure and relationships, not just individual data points. Graphs make these relationships explicit and analyzable.

## Formal Definition

A **graph** G is defined as an ordered pair G = (V, E) where:
- V is a set of **nodes** (also called vertices): V = {v₁, v₂, ..., vₙ}
- E is a set of **edges** (also called links): E ⊆ {(vᵢ, vⱼ) | vᵢ, vⱼ ∈ V}

An edge (vᵢ, vⱼ) represents a connection between nodes vᵢ and vⱼ.

**Types of graphs:**
- **Undirected graph**: Edges have no direction; (vᵢ, vⱼ) = (vⱼ, vᵢ). Example: Facebook friendships (mutual)
- **Directed graph** (digraph): Edges have direction; (vᵢ, vⱼ) ≠ (vⱼ, vᵢ). Example: Twitter follows (one-way)
- **Weighted graph**: Each edge has a weight w(vᵢ, vⱼ) ∈ ℝ. Example: Road networks with distances
- **Unweighted graph**: All edges are equivalent (weight = 1)

**Key properties:**
- **Degree** d(v): Number of edges connected to node v. For directed graphs: in-degree (incoming edges) and out-degree (outgoing edges)
- **Path**: Sequence of nodes v₁, v₂, ..., vₖ where consecutive nodes are connected by edges
- **Path length**: Number of edges in a path
- **Connected graph**: A path exists between every pair of nodes
- **Connected component**: Maximal set of nodes where every pair is connected by a path
- **Cycle**: A path that starts and ends at the same node
- **Diameter**: Length of the longest shortest path between any two nodes in the graph

**Graph representations:**

1. **Adjacency matrix** A: n × n matrix where Aᵢⱼ = 1 if edge (vᵢ, vⱼ) exists, 0 otherwise
   - For weighted graphs: Aᵢⱼ = w(vᵢ, vⱼ)
   - For undirected graphs: A is symmetric (Aᵢⱼ = Aⱼᵢ)
   - Space complexity: O(n²)

2. **Adjacency list**: For each node vᵢ, store a list of its neighbors
   - Space complexity: O(n + m) where m = |E|
   - Efficient for sparse graphs (m << n²)

3. **Edge list**: List of all edges [(v₁, v₂), (v₃, v₄), ...]
   - Simple but inefficient for queries
   - Space complexity: O(m)

**Graph density**: ρ = m / (n(n-1)/2) for undirected graphs, where m = |E| and n = |V|
- **Sparse graph**: ρ << 1 (few edges relative to possible edges)
- **Dense graph**: ρ → 1 (many edges, close to complete graph)

> **Key Concept:** A graph represents entities (nodes) and their relationships (edges), making it possible to analyze structure and connectivity in systems ranging from social networks to molecules.

## Visualization

The following diagram shows fundamental graph terminology with labeled examples:

```
[DIAGRAM: Graph Terminology Visual Dictionary]

Components shown:
1. Node/Vertex (circle labeled "A")
2. Edge/Link (line connecting two nodes)
3. Directed edge (arrow from A to B)
4. Weighted edge (line labeled with "5")
5. Self-loop (edge from node to itself)
6. Path (highlighted sequence A→B→C→D)
7. Cycle (path that returns to start: A→B→C→A)
8. Clique (complete subgraph: all nodes connected)

Side-by-side comparison of same 6-node graph in three representations:
- Left: Visual graph layout (nodes as circles, edges as lines)
- Middle: Adjacency matrix (6×6 grid with 0s and 1s)
- Right: Adjacency list (text format showing neighbors)
```

## Examples

### Part 1: Loading and Visualizing a Graph

```python
# Loading and visualizing the Karate Club graph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Load Zachary's Karate Club graph - classic social network dataset
# 34 members of a karate club, 78 friendships
G = nx.karate_club_graph()

# Print basic statistics
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
avg_degree = 2 * n_edges / n_nodes  # For undirected graphs
density = nx.density(G)

print("Zachary's Karate Club Network")
print("=" * 40)
print(f"Number of nodes (members): {n_nodes}")
print(f"Number of edges (friendships): {n_edges}")
print(f"Average degree: {avg_degree:.2f}")
print(f"Network density: {density:.3f}")
print()

# Visualize the graph
plt.figure(figsize=(12, 5))

# Panel 1: Graph visualization with spring layout
plt.subplot(1, 2, 1)
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                       node_size=300, alpha=0.9)
nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.5)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title("Karate Club Social Network", fontsize=12, fontweight='bold')
plt.axis('off')

# Panel 2: Degree distribution
plt.subplot(1, 2, 2)
degrees = [G.degree(n) for n in G.nodes()]
plt.hist(degrees, bins=range(1, max(degrees) + 2), edgecolor='black', alpha=0.7)
plt.xlabel('Degree (Number of Friends)', fontsize=10)
plt.ylabel('Frequency (Number of Members)', fontsize=10)
plt.title('Degree Distribution', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('karate_club_basic.png', dpi=150, bbox_inches='tight')
plt.show()

# Output:
# Zachary's Karate Club Network
# ========================================
# Number of nodes (members): 34
# Number of edges (friendships): 78
# Average degree: 4.59
# Network density: 0.139
```

The code loads Zachary's Karate Club graph, a famous dataset from social network research representing friendships among 34 members of a karate club. The graph has 78 edges (friendships) and an average degree of 4.59, meaning each member has about 4-5 friends on average. The density of 0.139 indicates this is a sparse network—only 13.9% of all possible friendships exist. The visualization shows the network layout using a spring layout algorithm that positions connected nodes closer together, revealing the community structure. The degree distribution histogram shows most members have 2-6 friends, with a few highly connected individuals.

### Part 2: Graph Representations

```python
# Representing graphs in different formats
# Adjacency matrix representation
A = nx.adjacency_matrix(G)  # Returns scipy sparse matrix
A_dense = A.todense()  # Convert to dense matrix for display

print("Adjacency Matrix Representation")
print("=" * 40)
print(f"Matrix shape: {A_dense.shape}")
print(f"Matrix type: Sparse CSR (memory efficient)")
print(f"Number of non-zero entries: {A.nnz}")
print(f"Sparsity: {1 - A.nnz / (n_nodes * n_nodes):.3f}")
print()

# Show small portion of adjacency matrix
print("First 10×10 block of adjacency matrix:")
print(A_dense[:10, :10])
print()

# Visualize adjacency matrix as heatmap
plt.figure(figsize=(10, 8))
plt.imshow(A_dense, cmap='Blues', interpolation='nearest')
plt.colorbar(label='Connection (1=edge exists, 0=no edge)')
plt.title('Adjacency Matrix Heatmap\n(Dark squares = friendships)',
          fontsize=12, fontweight='bold')
plt.xlabel('Node ID', fontsize=10)
plt.ylabel('Node ID', fontsize=10)
plt.tight_layout()
plt.savefig('adjacency_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Adjacency list representation
print("Adjacency List Representation")
print("=" * 40)
print("(Each node followed by its neighbors)")
print()

# Show adjacency list for first 5 nodes
for node in list(G.nodes())[:5]:
    neighbors = list(G.neighbors(node))
    print(f"Node {node:2d}: {neighbors}")
print("...")
print()

# Edge list representation
edge_list = list(G.edges())
print("Edge List Representation")
print("=" * 40)
print(f"Total edges: {len(edge_list)}")
print("First 10 edges:")
for i, (u, v) in enumerate(edge_list[:10]):
    print(f"  ({u:2d}, {v:2d})")
print("...")
print()

# Convert to pandas DataFrame for analysis
edge_df = pd.DataFrame(edge_list, columns=['source', 'target'])
print("Edge DataFrame:")
print(edge_df.head(10))

# Output:
# Adjacency Matrix Representation
# ========================================
# Matrix shape: (34, 34)
# Matrix type: Sparse CSR (memory efficient)
# Number of non-zero entries: 156
# Sparsity: 0.865
#
# First 10×10 block of adjacency matrix:
# [[0 1 1 1 1 1 1 1 1 0]
#  [1 0 1 1 0 0 0 1 0 0]
#  [1 1 0 1 0 0 0 1 1 1]
#  [1 1 1 0 0 0 0 1 0 0]
#  [1 0 0 0 0 0 1 0 0 0]
#  [1 0 0 0 0 0 1 0 0 0]
#  [1 0 0 0 1 1 0 0 0 0]
#  [1 1 1 1 0 0 0 0 0 0]
#  [1 0 1 0 0 0 0 0 0 0]
#  [0 0 1 0 0 0 0 0 0 0]]
#
# Adjacency List Representation
# ========================================
# (Each node followed by its neighbors)
#
# Node  0: [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31]
# Node  1: [0, 2, 3, 7, 13, 17, 19, 21, 30]
# Node  2: [0, 1, 3, 7, 8, 9, 13, 27, 28, 32]
# Node  3: [0, 1, 2, 7, 12, 13]
# Node  4: [0, 6, 10]
# ...
```

This code demonstrates the three main ways to represent graphs computationally. The **adjacency matrix** is a 34×34 matrix where entry (i,j) is 1 if nodes i and j are connected, 0 otherwise. It's symmetric for undirected graphs (Aᵢⱼ = Aⱼᵢ). With 86.5% of entries being zero (sparsity), storing it as a sparse matrix saves memory. The heatmap visualization shows the pattern of connections—dark squares indicate friendships.

The **adjacency list** stores each node with its neighbors. Node 0 (the karate club instructor) has 16 friends, while most nodes have fewer. This representation is memory-efficient for sparse graphs and fast for finding neighbors.

The **edge list** simply lists all edges as pairs. It's the most compact but requires scanning all edges to find a node's neighbors. Converting to a DataFrame makes it easy to filter and analyze edges.

Each representation has tradeoffs: adjacency matrices enable fast edge lookups (O(1)) but use O(n²) space; adjacency lists use O(n+m) space and are efficient for sparse graphs; edge lists are simple but slow for queries.

### Part 3: Computing Graph Properties

```python
# Computing fundamental graph properties
print("Graph Properties Analysis")
print("=" * 50)

# Degree analysis
degrees = dict(G.degree())
max_degree_node = max(degrees, key=degrees.get)
min_degree_node = min(degrees, key=degrees.get)

print("Degree Analysis:")
print(f"  Node with most friends (highest degree): {max_degree_node} (degree={degrees[max_degree_node]})")
print(f"  Node with fewest friends (lowest degree): {min_degree_node} (degree={degrees[min_degree_node]})")
print(f"  Average degree: {sum(degrees.values()) / len(degrees):.2f}")
print()

# Connectivity analysis
is_connected = nx.is_connected(G)
n_components = nx.number_connected_components(G)
components = list(nx.connected_components(G))

print("Connectivity:")
print(f"  Is connected? {is_connected}")
print(f"  Number of connected components: {n_components}")
if n_components > 1:
    print(f"  Component sizes: {[len(c) for c in components]}")
print()

# Path analysis
# Find shortest path between instructor (0) and administrator (33)
shortest_path = nx.shortest_path(G, source=0, target=33)
shortest_path_length = len(shortest_path) - 1

print("Path Analysis:")
print(f"  Shortest path from node 0 to node 33:")
print(f"    Path: {' → '.join(map(str, shortest_path))}")
print(f"    Length: {shortest_path_length} edges")
print()

# Diameter (longest shortest path)
diameter = nx.diameter(G)
avg_shortest_path = nx.average_shortest_path_length(G)

print(f"  Network diameter: {diameter}")
print(f"  Average shortest path length: {avg_shortest_path:.3f}")
print()

# Clustering coefficient (measure of triangles)
clustering_coeffs = nx.clustering(G)
avg_clustering = nx.average_clustering(G)

print("Clustering Analysis:")
print(f"  Average clustering coefficient: {avg_clustering:.3f}")
print(f"  (Probability that two friends of a node are also friends)")
print()
print(f"  Top 5 most clustered nodes:")
sorted_nodes = sorted(clustering_coeffs.items(), key=lambda x: x[1], reverse=True)
for node, coef in sorted_nodes[:5]:
    print(f"    Node {node:2d}: {coef:.3f}")
print()

# Triangle count
triangles = nx.triangles(G)
total_triangles = sum(triangles.values()) // 3  # Each triangle counted 3 times

print(f"  Total number of triangles: {total_triangles}")
print(f"  (Groups of 3 mutually connected friends)")

# Output:
# Graph Properties Analysis
# ==================================================
# Degree Analysis:
#   Node with most friends (highest degree): 0 (degree=16)
#   Node with fewest friends (lowest degree): 11 (degree=1)
#   Average degree: 4.59
#
# Connectivity:
#   Is connected? True
#   Number of connected components: 1
#
# Path Analysis:
#   Shortest path from node 0 to node 33:
#     Path: 0 → 31 → 33
#     Length: 2 edges
#
#   Network diameter: 5
#   Average shortest path length: 2.408
#
# Clustering Analysis:
#   Average clustering coefficient: 0.571
#   (Probability that two friends of a node are also friends)
#
#   Top 5 most clustered nodes:
#     Node  5: 1.000
#     Node  6: 1.000
#     Node 11: 1.000
#     Node 13: 1.000
#     Node 14: 1.000
#
#   Total number of triangles: 45
#   (Groups of 3 mutually connected friends)
```

This analysis reveals key structural properties of the network. Node 0 (the instructor) is the most connected with 16 friends, while node 11 has only 1 friend. The network is fully connected—every member can reach every other member through some chain of friendships.

The shortest path from instructor (0) to administrator (33) is just 2 edges (0→31→33), showing they're connected through one mutual friend. The diameter of 5 means the maximum "degrees of separation" between any two members is 5, and on average, members are only 2.4 steps apart. This is characteristic of "small-world" networks where paths between nodes are surprisingly short.

The average clustering coefficient of 0.571 is high, indicating strong community structure. When two people are both friends with a third person, there's a 57.1% chance they're also friends with each other. Several nodes have clustering coefficient 1.0, meaning all their friends are also friends with each other (forming complete subgraphs or cliques). The network contains 45 triangles—groups of three mutually connected members—which are building blocks of tightly-knit communities.

### Part 4: Different Types of Graphs

```python
# Creating and comparing different graph types
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Types of Graphs', fontsize=14, fontweight='bold')

# 1. Undirected Graph
ax = axes[0, 0]
G_undirected = nx.Graph()
G_undirected.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (2, 4)])
pos1 = nx.spring_layout(G_undirected, seed=42)
nx.draw(G_undirected, pos1, ax=ax, with_labels=True, node_color='lightblue',
        node_size=500, font_size=12, font_weight='bold', arrows=False)
ax.set_title('Undirected Graph\n(e.g., Facebook friendships)', fontsize=11)
ax.axis('off')

# 2. Directed Graph
ax = axes[0, 1]
G_directed = nx.DiGraph()
G_directed.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (2, 4)])
pos2 = nx.spring_layout(G_directed, seed=42)
nx.draw(G_directed, pos2, ax=ax, with_labels=True, node_color='lightcoral',
        node_size=500, font_size=12, font_weight='bold',
        arrows=True, arrowsize=20, arrowstyle='->')
ax.set_title('Directed Graph\n(e.g., Twitter follows)', fontsize=11)
ax.axis('off')

# 3. Weighted Graph
ax = axes[0, 2]
G_weighted = nx.Graph()
edges_with_weights = [(1, 2, 4), (2, 3, 2), (3, 4, 5), (4, 1, 3), (2, 4, 1)]
G_weighted.add_weighted_edges_from(edges_with_weights)
pos3 = nx.spring_layout(G_weighted, seed=42)
nx.draw(G_weighted, pos3, ax=ax, with_labels=True, node_color='lightgreen',
        node_size=500, font_size=12, font_weight='bold')
edge_labels = nx.get_edge_attributes(G_weighted, 'weight')
nx.draw_networkx_edge_labels(G_weighted, pos3, edge_labels, ax=ax, font_size=10)
ax.set_title('Weighted Graph\n(e.g., road distances)', fontsize=11)
ax.axis('off')

# 4. Bipartite Graph
ax = axes[1, 0]
G_bipartite = nx.Graph()
# Users (nodes 0-2) and movies (nodes 3-5)
G_bipartite.add_nodes_from([0, 1, 2], bipartite=0)  # Users
G_bipartite.add_nodes_from([3, 4, 5], bipartite=1)  # Movies
G_bipartite.add_edges_from([(0, 3), (0, 4), (1, 4), (1, 5), (2, 3), (2, 5)])
pos_bipartite = {0: (0, 1), 1: (0, 0), 2: (0, -1),
                 3: (2, 1), 4: (2, 0), 5: (2, -1)}
colors = ['lightblue' if G_bipartite.nodes[n].get('bipartite') == 0
          else 'lightyellow' for n in G_bipartite.nodes()]
nx.draw(G_bipartite, pos_bipartite, ax=ax, with_labels=True,
        node_color=colors, node_size=500, font_size=12, font_weight='bold')
ax.set_title('Bipartite Graph\n(e.g., users and movies)', fontsize=11)
ax.text(0, -1.5, 'Users', ha='center', fontsize=10, color='blue')
ax.text(2, -1.5, 'Movies', ha='center', fontsize=10, color='orange')
ax.axis('off')

# 5. Multigraph (multiple edges between nodes)
ax = axes[1, 1]
G_multi = nx.MultiGraph()
G_multi.add_edges_from([(1, 2), (1, 2), (2, 3), (3, 4), (4, 1)])
pos5 = nx.circular_layout(G_multi)
nx.draw(G_multi, pos5, ax=ax, with_labels=True, node_color='plum',
        node_size=500, font_size=12, font_weight='bold')
ax.set_title('Multigraph\n(multiple edges allowed)', fontsize=11)
ax.axis('off')

# 6. Tree (connected acyclic graph)
ax = axes[1, 2]
G_tree = nx.balanced_tree(2, 2)  # Binary tree, depth 2
pos6 = nx.spring_layout(G_tree, seed=42)
nx.draw(G_tree, pos6, ax=ax, with_labels=True, node_color='peachpuff',
        node_size=500, font_size=12, font_weight='bold')
ax.set_title('Tree\n(hierarchical structure)', fontsize=11)
ax.axis('off')

plt.tight_layout()
plt.savefig('graph_types.png', dpi=150, bbox_inches='tight')
plt.show()

print("Graph Type Comparisons")
print("=" * 50)

# Compare adjacency matrix properties
print("\nAdjacency Matrix Properties:")
print(f"Undirected - Symmetric? {np.allclose(nx.adjacency_matrix(G_undirected).todense(),
                                               nx.adjacency_matrix(G_undirected).todense().T)}")
print(f"Directed   - Symmetric? {np.allclose(nx.adjacency_matrix(G_directed).todense(),
                                               nx.adjacency_matrix(G_directed).todense().T)}")

# Check bipartite property
from networkx.algorithms import bipartite
is_bipartite = bipartite.is_bipartite(G_bipartite)
print(f"\nBipartite graph check: {is_bipartite}")

# Check tree properties
is_tree = nx.is_tree(G_tree)
print(f"Tree properties:")
print(f"  Is connected and acyclic? {is_tree}")
print(f"  Number of edges: {G_tree.number_of_edges()}")
print(f"  Number of nodes: {G_tree.number_of_nodes()}")
print(f"  Tree property: n_edges = n_nodes - 1? {G_tree.number_of_edges() == G_tree.number_of_nodes() - 1}")

# Output:
# Graph Type Comparisons
# ==================================================
#
# Adjacency Matrix Properties:
# Undirected - Symmetric? True
# Directed   - Symmetric? False
#
# Bipartite graph check: True
# Tree properties:
#   Is connected and acyclic? True
#   Number of edges: 6
#   Number of nodes: 7
#   Tree property: n_edges = n_nodes - 1? True
```

This example demonstrates six fundamental graph types. **Undirected graphs** have symmetric relationships (Facebook friendships are mutual), reflected in symmetric adjacency matrices. **Directed graphs** have asymmetric edges (A following B on Twitter doesn't mean B follows A), with potentially asymmetric adjacency matrices.

**Weighted graphs** assign numerical values to edges, like distances in road networks or similarity scores. **Bipartite graphs** have two distinct node types (users and movies) with edges only connecting different types—useful for recommendation systems and matchings. **Multigraphs** allow multiple edges between nodes (e.g., multiple flight routes between cities). **Trees** are connected acyclic graphs with exactly n-1 edges for n nodes—common in hierarchies and decision processes.

Understanding graph types is crucial for choosing appropriate algorithms and representations. Different domains naturally correspond to different graph types, and recognizing this structure guides analysis strategy.

## Common Pitfalls

**1. Using Dense Adjacency Matrices for Large Sparse Graphs**

Real-world networks are typically sparse—most possible edges don't exist. Storing a sparse graph as a dense n×n matrix wastes memory and slows computation. For a social network with 1 million users and 10 million friendships, a dense matrix requires 1 trillion entries (8TB of memory with double precision), while a sparse representation needs only 80MB.

**Solution:** Use sparse matrix representations (scipy.sparse or NetworkX's default). Operations like matrix multiplication and linear algebra work efficiently on sparse matrices without materializing zeros. The scipy CSR (Compressed Sparse Row) format is ideal for row-slicing and matrix multiplication.

**2. Confusing Undirected and Directed Semantics**

Treating a directed graph as undirected (or vice versa) changes analysis results. Computing PageRank on an undirected graph is meaningless because PageRank is defined by following directed links. Similarly, applying undirected community detection to a directed citation network ignores the asymmetry of citing vs. being cited.

**Solution:** Always clarify whether relationships are symmetric or asymmetric before analysis. Many NetworkX algorithms have separate versions for directed and undirected graphs (e.g., `connected_components` vs. `strongly_connected_components`). Check algorithm assumptions about graph type.

**3. Forgetting Disconnected Components**

Algorithms like shortest path and diameter assume a connected graph. If the graph has multiple disconnected components, these functions either fail or give misleading results. Computing the diameter of a disconnected graph raises an exception because no path exists between components.

**Solution:** Always check connectivity first using `nx.is_connected(G)` for undirected graphs or `nx.is_strongly_connected(G)` for directed graphs. If disconnected, either analyze the largest connected component (`max(nx.connected_components(G), key=len)`) or compute properties per component separately.

## Practice Exercises

**Exercise 1**

Create a graph representing relationships among 10 people with 20 friendships. Use NetworkX to build the graph, visualize it, and compute:
- The person with the most friends (highest degree)
- The average number of friends
- The shortest path between any two specific people
- Whether the graph is connected (everyone reachable from everyone else)

Interpret the results: who is most "central"? What does the diameter tell you about the group's cohesion?

**Exercise 2**

Load the Les Misérables character co-appearance network using `G = nx.les_miserables_graph()`. This graph has 77 characters with edges weighted by number of scenes they appear together in. Analyze:
- Top 5 characters by degree (most connected)
- Distribution of edge weights (visualize with histogram)
- Average clustering coefficient compared to a random graph with the same number of nodes and edges
- Network diameter and average shortest path length

Interpret: Which characters are central to the story? Is this network more clustered than random? What does clustering reveal about the story structure?

**Exercise 3**

Build a directed citation network representing 20 papers where some papers cite others. You can create this manually or use a subset of a real citation dataset. For your network, compute:
- In-degree distribution (papers with most citations)
- Out-degree distribution (papers that cite many others)
- Papers with no incoming edges (not cited—possibly new papers or low-impact)
- Papers with no outgoing edges (cite nothing—possibly review papers)

Visualize the graph with node sizes proportional to in-degree. Discuss: What patterns do citation networks exhibit? How does the in-degree distribution differ from the out-degree distribution?

**Exercise 4**

Compare how different graph representations scale. Create random graphs of increasing size (n = 100, 1000, 10000 nodes) with density ρ = 0.01 (sparse). For each graph:
- Measure memory usage of dense adjacency matrix (numpy array) vs. sparse matrix (scipy.sparse.csr_matrix)
- Time how long it takes to find all neighbors of a node using: (a) adjacency matrix, (b) adjacency list, (c) edge list
- Plot memory usage and query time vs. graph size

Interpret: At what graph size does dense representation become impractical? Which representation is fastest for neighbor queries?

**Exercise 5**

Investigate the "small-world" property. Generate three types of graphs with n=100 nodes:
1. Random graph (Erdős-Rényi) with p=0.05: `nx.erdos_renyi_graph(100, 0.05, seed=42)`
2. Small-world graph (Watts-Strogatz) with k=4, p=0.3: `nx.watts_strogatz_graph(100, 4, 0.3, seed=42)`
3. Regular ring lattice with k=4: `nx.watts_strogatz_graph(100, 4, 0, seed=42)` (p=0 means no rewiring)

For each graph, compute:
- Average shortest path length
- Average clustering coefficient

Small-world networks have short paths (like random graphs) and high clustering (like regular lattices). Which of your graphs exhibits small-world properties? How common are small-world networks in real applications?

## Solutions

**Solution 1**

```python
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Set random seed
np.random.seed(42)

# Create graph with 10 people
G = nx.Graph()
people = list(range(10))
G.add_nodes_from(people)

# Add 20 random friendships (ensuring no self-loops or duplicates)
edges = set()
while len(edges) < 20:
    u, v = np.random.choice(people, size=2, replace=False)
    if u != v:
        edges.add((min(u, v), max(u, v)))

G.add_edges_from(edges)

# Compute properties
degrees = dict(G.degree())
most_friends = max(degrees, key=degrees.get)
avg_friends = sum(degrees.values()) / len(degrees)

# Shortest path between person 0 and person 9
if nx.has_path(G, 0, 9):
    path = nx.shortest_path(G, 0, 9)
    path_length = len(path) - 1
else:
    path = None
    path_length = float('inf')

# Connectivity
is_connected = nx.is_connected(G)
if is_connected:
    diameter = nx.diameter(G)
else:
    diameter = "Undefined (graph disconnected)"

# Visualize
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
pos = nx.spring_layout(G, seed=42)
node_colors = ['red' if n == most_friends else 'lightblue' for n in G.nodes()]
nx.draw(G, pos, with_labels=True, node_color=node_colors,
        node_size=500, font_size=12, font_weight='bold')
plt.title(f'Social Network\n(Red node = most friends)', fontsize=11)

plt.subplot(1, 2, 2)
plt.bar(degrees.keys(), degrees.values(), color='skyblue', edgecolor='black')
plt.xlabel('Person ID')
plt.ylabel('Number of Friends')
plt.title('Degree Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Person with most friends: {most_friends} ({degrees[most_friends]} friends)")
print(f"Average number of friends: {avg_friends:.2f}")
if path:
    print(f"Shortest path from 0 to 9: {' → '.join(map(str, path))} (length {path_length})")
else:
    print("No path exists between person 0 and person 9 (disconnected)")
print(f"Graph connected? {is_connected}")
if isinstance(diameter, int):
    print(f"Network diameter: {diameter}")
else:
    print(f"Diameter: {diameter}")

# Interpretation
print("\nInterpretation:")
print(f"Person {most_friends} is most central by degree centrality (has most direct connections).")
if is_connected and isinstance(diameter, int):
    if diameter <= 3:
        print(f"The diameter of {diameter} suggests a tightly-knit group where everyone is close.")
    else:
        print(f"The diameter of {diameter} suggests some people are separated by multiple intermediaries.")
```

**Interpretation:** The person with highest degree is most directly connected, serving as a hub. A small diameter (≤3) indicates high cohesion—everyone can reach everyone else quickly. If the graph is disconnected, it reveals isolated subgroups who don't interact.

**Solution 2**

```python
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Load Les Misérables graph
G = nx.les_miserables_graph()

# Top 5 characters by degree
degrees = dict(G.degree())
top5 = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]

print("Top 5 characters by degree (most connections):")
for char, deg in top5:
    print(f"  {char:20s}: {deg} connections")

# Edge weight distribution
weights = [G[u][v]['weight'] for u, v in G.edges()]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(weights, bins=20, edgecolor='black', alpha=0.7, color='coral')
plt.xlabel('Edge Weight (Co-appearances)')
plt.ylabel('Frequency')
plt.title('Distribution of Co-appearance Weights')
plt.grid(True, alpha=0.3)

# Clustering coefficient comparison
avg_clustering_lesmis = nx.average_clustering(G)

# Generate random graph with same n and m
n = G.number_of_nodes()
m = G.number_of_edges()
p = 2 * m / (n * (n - 1))  # Probability for Erdős-Rényi
G_random = nx.erdos_renyi_graph(n, p, seed=42)
avg_clustering_random = nx.average_clustering(G_random)

print(f"\nClustering coefficient:")
print(f"  Les Misérables: {avg_clustering_lesmis:.3f}")
print(f"  Random graph:   {avg_clustering_random:.3f}")
print(f"  Ratio (Les Mis / Random): {avg_clustering_lesmis / avg_clustering_random:.1f}x")

# Diameter and average path length
diameter = nx.diameter(G)
avg_path = nx.average_shortest_path_length(G)

print(f"\nPath metrics:")
print(f"  Diameter: {diameter}")
print(f"  Average shortest path length: {avg_path:.3f}")

# Visualization
plt.subplot(1, 2, 2)
pos = nx.spring_layout(G, seed=42, k=0.5)
node_sizes = [degrees[node] * 30 for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.6, node_color='lightblue')
nx.draw_networkx_edges(G, pos, alpha=0.1)
nx.draw_networkx_labels(G, pos, font_size=6)
plt.title('Les Misérables Character Network\n(Node size ∝ degree)')
plt.axis('off')

plt.tight_layout()
plt.show()

print("\nInterpretation:")
print("The top-degree characters are main characters (Valjean, Myriel, etc.).")
print("Clustering is much higher than random, indicating tight-knit groups (communities).")
print("Short average path length shows characters are closely connected narratively.")
```

**Interpretation:** Central characters like Jean Valjean have many connections. The clustering coefficient being ~6x higher than random reveals strong community structure—groups of characters who interact mainly with each other. The short path length despite high clustering is the "small-world" property characteristic of social networks.

**Solution 3**

```python
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Create directed citation network (20 papers)
np.random.seed(42)
G = nx.DiGraph()
papers = list(range(20))
G.add_nodes_from(papers)

# Add directed edges (citations): newer papers cite older papers
# Bias: papers are more likely to cite earlier papers
edges = []
for paper in range(5, 20):  # Papers 5-19 cite earlier papers
    n_citations = np.random.randint(1, 6)  # Each paper cites 1-5 others
    cited = np.random.choice(range(0, paper), size=n_citations, replace=False)
    for c in cited:
        edges.append((paper, c))  # paper cites c

G.add_edges_from(edges)

# Compute in-degree and out-degree
in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())

# Papers not cited (in-degree = 0)
not_cited = [p for p in papers if in_degrees[p] == 0]

# Papers that cite nothing (out-degree = 0)
cite_nothing = [p for p in papers if out_degrees[p] == 0]

print("Citation Network Analysis")
print("=" * 50)
print(f"Total papers: {G.number_of_nodes()}")
print(f"Total citations: {G.number_of_edges()}")

print("\nIn-degree distribution (times cited):")
sorted_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
for paper, deg in sorted_in[:5]:
    print(f"  Paper {paper:2d}: cited {deg} times")

print("\nOut-degree distribution (papers cited):")
sorted_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
for paper, deg in sorted_out[:5]:
    print(f"  Paper {paper:2d}: cites {deg} papers")

print(f"\nPapers never cited (in-degree=0): {not_cited}")
print(f"Papers citing nothing (out-degree=0): {cite_nothing}")

# Visualize with node size proportional to in-degree
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
pos = nx.spring_layout(G, seed=42)
node_sizes = [max(100, in_degrees[n] * 100) for n in G.nodes()]
node_colors = [in_degrees[n] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                       cmap='YlOrRd', alpha=0.8, vmin=0)
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray',
                       arrows=True, arrowsize=10, arrowstyle='->')
nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
plt.title('Citation Network\n(Node size ∝ citations received)', fontsize=11)
plt.axis('off')

# In-degree distribution
plt.subplot(1, 2, 2)
in_deg_values = list(in_degrees.values())
plt.hist(in_deg_values, bins=range(0, max(in_deg_values)+2),
         edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('In-Degree (Citations Received)')
plt.ylabel('Number of Papers')
plt.title('In-Degree Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nInterpretation:")
print("Citation networks typically follow a power-law distribution:")
print("- Most papers receive few citations (long tail)")
print("- Few papers receive many citations (high-impact papers)")
print("- Papers with in-degree=0 are recent or low-impact")
print("- Papers with out-degree=0 may be foundational (others cite them)")
print(f"In-degree distribution has high variance: mean={np.mean(in_deg_values):.1f}, "
      f"max={max(in_deg_values)}")
```

**Interpretation:** Citation networks exhibit "preferential attachment"—highly-cited papers attract more citations ("rich get richer"). The in-degree distribution is right-skewed with many low-citation papers and few highly-cited ones. Papers with zero out-degree are either foundational (cited but cite nothing) or isolated. Papers with zero in-degree are recent or unrecognized.

**Solution 4**

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import csr_matrix
import sys

# Generate random sparse graphs of increasing size
sizes = [100, 1000, 5000, 10000]
density = 0.01

results = {
    'size': [],
    'dense_memory_mb': [],
    'sparse_memory_mb': [],
    'adj_matrix_time_ms': [],
    'adj_list_time_ms': []
}

for n in sizes:
    print(f"Testing graph with {n} nodes...")

    # Generate random graph
    m = int(n * (n - 1) * density / 2)  # Number of edges
    G = nx.gnm_random_graph(n, m, seed=42)

    # Dense adjacency matrix memory
    A_dense = nx.to_numpy_array(G)
    dense_mem = A_dense.nbytes / (1024 ** 2)  # MB

    # Sparse adjacency matrix memory
    A_sparse = nx.adjacency_matrix(G)  # scipy sparse CSR
    sparse_mem = (A_sparse.data.nbytes + A_sparse.indices.nbytes +
                  A_sparse.indptr.nbytes) / (1024 ** 2)  # MB

    # Time to find all neighbors using adjacency matrix
    start = time.time()
    for _ in range(100):
        node = np.random.randint(0, n)
        neighbors_matrix = np.where(A_dense[node] > 0)[0]
    adj_matrix_time = (time.time() - start) * 1000 / 100  # ms per query

    # Time to find all neighbors using adjacency list (NetworkX's default)
    start = time.time()
    for _ in range(100):
        node = np.random.randint(0, n)
        neighbors_list = list(G.neighbors(node))
    adj_list_time = (time.time() - start) * 1000 / 100  # ms per query

    results['size'].append(n)
    results['dense_memory_mb'].append(dense_mem)
    results['sparse_memory_mb'].append(sparse_mem)
    results['adj_matrix_time_ms'].append(adj_matrix_time)
    results['adj_list_time_ms'].append(adj_list_time)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Memory usage
axes[0].plot(results['size'], results['dense_memory_mb'],
             'o-', label='Dense Matrix', linewidth=2, markersize=8)
axes[0].plot(results['size'], results['sparse_memory_mb'],
             's-', label='Sparse Matrix', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Nodes', fontsize=11)
axes[0].set_ylabel('Memory Usage (MB)', fontsize=11)
axes[0].set_title('Memory Scaling for Sparse Graphs (density=0.01)', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# Query time
axes[1].plot(results['size'], results['adj_matrix_time_ms'],
             'o-', label='Adjacency Matrix', linewidth=2, markersize=8)
axes[1].plot(results['size'], results['adj_list_time_ms'],
             's-', label='Adjacency List', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Nodes', fontsize=11)
axes[1].set_ylabel('Average Query Time (ms)', fontsize=11)
axes[1].set_title('Neighbor Query Time', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nResults Summary:")
print("=" * 70)
print(f"{'Nodes':>8s} | {'Dense (MB)':>12s} | {'Sparse (MB)':>12s} | "
      f"{'Ratio':>8s} | {'Matrix (ms)':>12s} | {'List (ms)':>10s}")
print("-" * 70)
for i in range(len(results['size'])):
    ratio = results['dense_memory_mb'][i] / results['sparse_memory_mb'][i]
    print(f"{results['size'][i]:8d} | {results['dense_memory_mb'][i]:12.2f} | "
          f"{results['sparse_memory_mb'][i]:12.2f} | {ratio:8.1f}x | "
          f"{results['adj_matrix_time_ms'][i]:12.4f} | "
          f"{results['adj_list_time_ms'][i]:10.4f}")

print("\nInterpretation:")
print(f"- Dense matrix memory grows as O(n²) - becomes impractical beyond n≈10k")
print(f"- Sparse matrix memory grows as O(n+m) - scales to millions of nodes")
print(f"- For this sparse graph (density=0.01), sparse saves {ratio:.0f}x memory")
print(f"- Adjacency list is faster for neighbor queries when degree << n")
print(f"- Dense matrix is O(n) per query; adjacency list is O(degree)")
```

**Interpretation:** Dense matrices scale quadratically—a 10,000-node graph requires 800MB even with just 1% density. Sparse matrices scale linearly with edges, saving 100x+ memory. For neighbor queries, adjacency lists are faster when average degree is small (sparse graphs), while dense matrices are faster for dense graphs where most entries are nonzero. Use sparse representations for real-world networks.

**Solution 5**

```python
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Set random seed
np.random.seed(42)

# Generate three graph types
n = 100

G_random = nx.erdos_renyi_graph(n, 0.05, seed=42)
G_smallworld = nx.watts_strogatz_graph(n, 4, 0.3, seed=42)
G_regular = nx.watts_strogatz_graph(n, 4, 0.0, seed=42)

graphs = {
    'Random (Erdős-Rényi)': G_random,
    'Small-World (Watts-Strogatz)': G_smallworld,
    'Regular Ring Lattice': G_regular
}

results = {}

for name, G in graphs.items():
    # Ensure connected for diameter computation
    if nx.is_connected(G):
        avg_path = nx.average_shortest_path_length(G)
    else:
        # Use largest component
        largest_cc = max(nx.connected_components(G), key=len)
        G_sub = G.subgraph(largest_cc).copy()
        avg_path = nx.average_shortest_path_length(G_sub)

    avg_clustering = nx.average_clustering(G)

    results[name] = {
        'avg_path': avg_path,
        'avg_clustering': avg_clustering
    }

# Display results
print("Small-World Property Analysis")
print("=" * 60)
print(f"{'Graph Type':<30s} | {'Avg Path Length':>15s} | {'Avg Clustering':>15s}")
print("-" * 60)

for name, metrics in results.items():
    print(f"{name:<30s} | {metrics['avg_path']:15.3f} | {metrics['avg_clustering']:15.3f}")

print("\nSmall-World Criteria:")
print("- Short average path length (like random graphs)")
print("- High clustering coefficient (like regular lattices)")

print("\nInterpretation:")
random_path = results['Random (Erdős-Rényi)']['avg_path']
random_clust = results['Random (Erdős-Rényi)']['avg_clustering']
sw_path = results['Small-World (Watts-Strogatz)']['avg_path']
sw_clust = results['Small-World (Watts-Strogatz)']['avg_clustering']
reg_path = results['Regular Ring Lattice']['avg_path']
reg_clust = results['Regular Ring Lattice']['avg_clustering']

print(f"\nRandom graph: short paths ({random_path:.2f}) but low clustering ({random_clust:.3f})")
print(f"Regular lattice: high clustering ({reg_clust:.3f}) but long paths ({reg_path:.2f})")
print(f"Small-world: combines both! Short paths ({sw_path:.2f}) AND high clustering ({sw_clust:.3f})")

print("\nSmall-world networks are common in:")
print("- Social networks (friend groups with occasional long-range connections)")
print("- Neural networks (local clusters with long-range connections)")
print("- Power grids (local distribution with long-range transmission lines)")
print("- The internet (local clusters of servers with backbone connections)")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, G) in zip(axes, graphs.items()):
    pos = nx.circular_layout(G) if 'Regular' in name or 'Small' in name else nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, node_size=30, node_color='lightblue',
            edge_color='gray', alpha=0.6, width=0.5)
    metrics = results[name]
    ax.set_title(f"{name}\nL={metrics['avg_path']:.2f}, C={metrics['avg_clustering']:.3f}",
                 fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()
```

**Interpretation:** The small-world graph combines the best of both worlds—short paths like random graphs (enabling fast information spread) and high clustering like regular lattices (enabling robust local communities). This property is ubiquitous in real networks: social networks have friend groups (high clustering) but you can reach strangers through a short chain (short paths). The "six degrees of separation" phenomenon is a manifestation of small-world structure.

## Key Takeaways

- A graph G=(V,E) consists of nodes (entities) and edges (relationships), providing a mathematical framework for analyzing networked systems from social networks to molecules
- Graph representations include adjacency matrices (O(n²) space, fast edge lookup), adjacency lists (O(n+m) space, efficient for sparse graphs), and edge lists (compact but slow queries)
- Key graph properties—degree, paths, connected components, diameter, and clustering coefficient—reveal structural patterns like hubs, communities, and small-world effects
- Different graph types (directed/undirected, weighted/unweighted, bipartite, trees) model different real-world relationships, and choosing the right type is crucial for meaningful analysis
- Real-world networks are typically sparse (density << 1), requiring sparse matrix representations to scale to millions of nodes efficiently

## Key Takeaways

**Next:** Section 38.2 covers network analysis techniques including centrality measures (degree, betweenness, closeness, eigenvector, PageRank) and community detection algorithms (Louvain, Girvan-Newman, label propagation) to identify important nodes and groups in networks.
