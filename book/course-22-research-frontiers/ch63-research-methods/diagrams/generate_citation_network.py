import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Create a citation graph for a research area (Transformers)
G = nx.DiGraph()

# Define papers (node attributes: title, year, citations count)
papers = {
    'RNN-Encoder-Decoder': {'year': 2014, 'citations': 8500, 'short': 'RNN Seq2Seq'},
    'Attention-Mechanism': {'year': 2015, 'citations': 15000, 'short': 'Attention'},
    'Transformer': {'year': 2017, 'citations': 95000, 'short': 'Transformer'},
    'BERT': {'year': 2019, 'citations': 75000, 'short': 'BERT'},
    'GPT-2': {'year': 2019, 'citations': 25000, 'short': 'GPT-2'},
    'GPT-3': {'year': 2020, 'citations': 45000, 'short': 'GPT-3'},
    'T5': {'year': 2020, 'citations': 18000, 'short': 'T5'},
    'Vision-Transformer': {'year': 2021, 'citations': 32000, 'short': 'ViT'},
    'Stable-Diffusion': {'year': 2022, 'citations': 12000, 'short': 'Stable Diff'},
    'GPT-4': {'year': 2023, 'citations': 8000, 'short': 'GPT-4'},
}

# Add nodes
for paper_id, attrs in papers.items():
    G.add_node(paper_id, **attrs)

# Add citation edges (who cites whom)
citations = [
    ('Transformer', 'RNN-Encoder-Decoder'),
    ('Transformer', 'Attention-Mechanism'),
    ('BERT', 'Transformer'),
    ('GPT-2', 'Transformer'),
    ('GPT-3', 'GPT-2'),
    ('GPT-3', 'BERT'),
    ('T5', 'Transformer'),
    ('T5', 'BERT'),
    ('Vision-Transformer', 'Transformer'),
    ('Stable-Diffusion', 'Transformer'),
    ('GPT-4', 'GPT-3'),
    ('GPT-4', 'Transformer'),
]

G.add_edges_from(citations)

# Compute layout
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Create visualization
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Node sizes proportional to citation count (influence)
node_sizes = [G.nodes[node]['citations'] / 50 for node in G.nodes()]

# Node colors by year (temporal progression)
years = [G.nodes[node]['year'] for node in G.nodes()]
node_colors = years

# Draw edges first (background)
nx.draw_networkx_edges(G, pos,
                       edge_color='gray',
                       arrows=True,
                       arrowsize=15,
                       arrowstyle='-|>',
                       width=1.5,
                       alpha=0.5,
                       connectionstyle='arc3,rad=0.1',
                       ax=ax)

# Draw nodes
nodes = nx.draw_networkx_nodes(G, pos,
                               node_size=node_sizes,
                               node_color=node_colors,
                               cmap='YlOrRd',
                               alpha=0.9,
                               edgecolors='black',
                               linewidths=2,
                               ax=ax)

# Draw labels
labels = {node: G.nodes[node]['short'] for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels,
                        font_size=9,
                        font_weight='bold',
                        font_color='white',
                        ax=ax)

# Add year annotations
for node, (x, y) in pos.items():
    year = G.nodes[node]['year']
    ax.text(x, y - 0.12, f"{year}",
           fontsize=7, ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))

# Colorbar for years
sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                           norm=plt.Normalize(vmin=min(years), vmax=max(years)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Publication Year', shrink=0.8)

ax.set_title('Citation Network: Evolution of Transformer Architecture\n' +
             'Node size = citation count (influence) | Edges = citation relationships',
             fontsize=14, weight='bold', pad=20)
ax.axis('off')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-22/ch63/diagrams/citation_network.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ citation_network.png saved")
