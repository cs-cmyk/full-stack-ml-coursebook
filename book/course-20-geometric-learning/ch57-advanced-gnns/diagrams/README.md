# Chapter 57 Diagrams

This directory contains all diagrams for Chapter 57: Advanced Graph Neural Networks

## Generated Diagrams

1. **message_passing_visualization.png** (79 KB)
   - Shows 3 stages of message passing: initial features, after 1st aggregation, after 2nd aggregation
   - Demonstrates how information propagates through graph structure
   - Illustrates the over-smoothing problem

2. **architecture_comparison.png** (162 KB)
   - Visual comparison of 4 GNN architectures: GCN, GraphSAGE, GAT, GIN
   - Shows different aggregation strategies (uniform, sampling, attention, sum+MLP)
   - Highlights key properties of each architecture

3. **architecture_performance_comparison.png** (51 KB)
   - Bar chart comparing test accuracy on Cora dataset
   - Shows GCN (0.801), GraphSAGE (0.814), GAT (0.823), GIN (0.818)
   - Uses color palette: blue, orange, green, red

4. **heterogeneous_graph_example.png** (322 KB)
   - Movie-Actor-Director heterogeneous graph
   - Shows multiple node types (movies, actors, directors) and edge types (stars_in, directs)
   - Demonstrates structure of heterogeneous graphs

5. **sampling_strategy_comparison.png** (123 KB)
   - Two-panel comparison of full-batch vs neighbor sampling
   - Left: exponential growth without sampling (1, 50, 2500, 125000, 6250000 nodes)
   - Right: controlled growth with sampling (1, 10, 100, 1000, 10000 nodes)
   - Log scale visualization

6. **sampling_tradeoffs.png** (98 KB)
   - Trade-off analysis: accuracy vs training time, accuracy vs memory usage
   - Compares Neighbor Sampling, Cluster-GCN, GraphSAINT
   - Shows production deployment recommendations

## Color Palette Used

- Blue (#2196F3): Primary nodes, Neighbor Sampling
- Green (#4CAF50): Secondary nodes, GraphSAINT, positive elements
- Orange (#FF9800): Directors, Cluster-GCN
- Red (#F44336): Error/warning elements, full-batch
- Purple (#9C27B0): (available for future use)
- Gray (#607D8B): Inactive/faded elements

## Source Files

Each diagram has a corresponding .py file that generates it:
- message_passing_visualization.py
- architecture_comparison.py
- architecture_performance_comparison.py
- heterogeneous_graph_example.py
- sampling_strategy_comparison.py
- sampling_tradeoffs.py

All diagrams are generated at 150 DPI with white backgrounds for textbook clarity.
