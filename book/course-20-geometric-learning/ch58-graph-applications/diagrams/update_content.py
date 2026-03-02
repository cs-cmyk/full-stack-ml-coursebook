"""
Update content.md to include diagram references
"""

import re

# Read the original content
with open('/home/chirag/ds-book/book/course-20/ch58/content.md', 'r') as f:
    content = f.read()

# Diagram insertions with their locations and text

# 1. Add domain overview diagram before the mermaid chart
diagram1_insert = '''## Visualization

![Domain Overview](diagrams/domain_overview.png)

**Figure 1**: Overview of graph ML application domains. Each domain has characteristic graph structures and task formulations. The five key application areas—molecular property prediction, fraud detection, knowledge graphs, recommendations, and traffic forecasting—each require different graph architectures and learning approaches.

```mermaid'''

content = content.replace('## Visualization\n\n```mermaid', diagram1_insert)

# 2. Update the mermaid chart caption to Figure 2
content = content.replace(
    '**Figure 1**: Overview of graph ML application domains. Each domain has characteristic graph structures and task formulations. Choosing the right architecture depends on graph type (homogeneous vs. heterogeneous), task (node vs. graph-level), and evaluation requirements.',
    '**Figure 2**: Detailed mapping of application domains to graph structures and tasks. Choosing the right architecture depends on graph type (homogeneous vs. heterogeneous), task (node vs. graph-level), and evaluation requirements.'
)

# 3. Add molecular workflow diagram after "## Examples"
examples_section = '## Examples\n\n### Part 1: Molecular Property Prediction with Message Passing'
molecular_workflow = '''## Examples

![Molecular Workflow](diagrams/molecular_workflow.png)

**Figure 3**: Molecular property prediction workflow. (1) Molecules are represented as graphs with atoms as nodes and bonds as edges. (2) Message passing aggregates information from neighboring atoms along chemical bonds. (3) Global pooling combines node embeddings into a graph-level representation for property prediction.

### Part 1: Molecular Property Prediction with Message Passing'''

content = content.replace(examples_section, molecular_workflow)

# 4. Add fraud detection diagram before "### Part 5: Fraud Detection with Heterogeneous Graphs"
fraud_section = '### Part 5: Fraud Detection with Heterogeneous Graphs'
fraud_diagram = '''![Fraud Detection Graph](diagrams/fraud_detection_graph.png)

**Figure 4**: Fraud detection using heterogeneous graphs. Fraudulent users (red circles) form a ring structure with circular money transfers, sharing devices and merchants. The graph structure reveals collusion patterns invisible to tabular models examining transactions independently.

### Part 5: Fraud Detection with Heterogeneous Graphs'''

content = content.replace(fraud_section, fraud_diagram)

# 5. Add knowledge graph diagram before "### Part 7: Knowledge Graph Embeddings with TransE"
kg_section = '### Part 7: Knowledge Graph Embeddings with TransE'
kg_diagram = '''![Knowledge Graph Embeddings](diagrams/knowledge_graph_embeddings.png)

**Figure 5**: TransE knowledge graph embeddings. Left: Triples represent facts as (head, relation, tail). Right: TransE learns entity and relation embeddings where h + r ≈ t in embedding space, enabling link prediction through vector arithmetic.

### Part 7: Knowledge Graph Embeddings with TransE'''

content = content.replace(kg_section, kg_diagram)

# 6. Add recommendation diagram before "### Part 9: Graph-Based Recommendation with LightGCN"
rec_section = '### Part 9: Graph-Based Recommendation with LightGCN'
rec_diagram = '''![Recommendation Bipartite Graph](diagrams/recommendation_bipartite.png)

**Figure 6**: Bipartite user-item graph for recommendations. Users (circles) and items (squares) are connected by interactions (edge thickness shows rating strength). Message passing propagates preferences through the graph, solving cold-start problems through collaborative filtering.

### Part 9: Graph-Based Recommendation with LightGCN'''

content = content.replace(rec_section, rec_diagram)

# 7. Add performance comparison diagram before "## Common Pitfalls"
pitfalls_section = '## Common Pitfalls'
performance_diagram = '''![Performance Comparison](diagrams/performance_comparison.png)

**Figure 7**: Performance comparison across domains. Top: GNN vs baseline metrics showing consistent improvements. GNNs excel when graph structure provides signal: molecular prediction (9% improvement), fraud detection (90% improvement from relational patterns), knowledge graphs (33% improvement from multi-hop reasoning), and recommendations (17% improvement for cold-start).

## Common Pitfalls'''

content = content.replace(pitfalls_section, performance_diagram)

# 8. Add decision tree diagram before "## Practice Exercises"
exercises_section = '## Practice Exercises'
decision_diagram = '''![Decision Tree](diagrams/decision_tree.png)

**Figure 8**: Application selection guide. Use this decision tree to determine when graph ML is appropriate and which architecture to choose based on your task (node classification, graph classification, or link prediction) and domain constraints.

## Practice Exercises'''

content = content.replace(exercises_section, decision_diagram)

# Write updated content
with open('/home/chirag/ds-book/book/course-20/ch58/content.md', 'w') as f:
    f.write(content)

print("✓ Updated content.md with diagram references")
print("\nSummary of changes:")
print("  - Added Figure 1: Domain Overview")
print("  - Updated mermaid chart to Figure 2")
print("  - Added Figure 3: Molecular Workflow")
print("  - Added Figure 4: Fraud Detection Graph")
print("  - Added Figure 5: Knowledge Graph Embeddings")
print("  - Added Figure 6: Recommendation Bipartite Graph")
print("  - Added Figure 7: Performance Comparison")
print("  - Added Figure 8: Decision Tree")
