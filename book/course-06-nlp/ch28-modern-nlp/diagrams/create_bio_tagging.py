"""
Create visualization of BIO tagging scheme for Named Entity Recognition
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.style.use('default')

# Create figure
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Example sentence
sentence = "Tim Cook works at Apple in California"
tokens = sentence.split()
tags = ["B-PER", "I-PER", "O", "O", "B-ORG", "O", "B-LOC"]

# Colors for different entity types
colors = {
    'PER': '#2196F3',  # Blue
    'ORG': '#4CAF50',  # Green
    'LOC': '#F44336',  # Red
    'O': '#607D8B'     # Gray
}

# Draw tokens and tags
x_start = 0.5
y_token = 3.5
y_tag = 1.5

for i, (token, tag) in enumerate(zip(tokens, tags)):
    # Determine entity type
    entity_type = tag.split('-')[-1] if tag != 'O' else 'O'
    color = colors[entity_type]

    # Token box
    token_box = FancyBboxPatch((x_start + i * 2, y_token - 0.3), 1.6, 0.6,
                               boxstyle="round,pad=0.05",
                               facecolor='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(token_box)
    ax.text(x_start + i * 2 + 0.8, y_token, token,
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Tag box
    tag_alpha = 0.8 if tag != 'O' else 0.3
    tag_box = FancyBboxPatch((x_start + i * 2, y_tag - 0.3), 1.6, 0.6,
                             boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black',
                             linewidth=1.5, alpha=tag_alpha)
    ax.add_patch(tag_box)
    ax.text(x_start + i * 2 + 0.8, y_tag, tag,
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='white' if tag != 'O' else 'black')

    # Connecting arrow
    arrow = FancyArrowPatch((x_start + i * 2 + 0.8, y_token - 0.4),
                           (x_start + i * 2 + 0.8, y_tag + 0.4),
                           arrowstyle='->', mutation_scale=15,
                           color='black', linewidth=1.5, alpha=0.5)
    ax.add_patch(arrow)

# Add title
ax.text(7, 5.2, 'BIO Tagging Scheme for Named Entity Recognition',
        ha='center', va='center', fontsize=14, fontweight='bold')

# Add labels
ax.text(7, 4.5, 'Input Tokens',
        ha='center', va='center', fontsize=11, style='italic', color='#666')
ax.text(7, 2.5, 'BIO Tags',
        ha='center', va='center', fontsize=11, style='italic', color='#666')

# Add legend with explanations
legend_y = 0.5
ax.text(0.5, legend_y, 'Legend:',
        ha='left', va='center', fontsize=11, fontweight='bold')

# B- tags
b_box = FancyBboxPatch((2.5, legend_y - 0.25), 0.5, 0.5,
                       boxstyle="round,pad=0.05",
                       facecolor=colors['PER'], edgecolor='black',
                       linewidth=1.5, alpha=0.8)
ax.add_patch(b_box)
ax.text(3.3, legend_y, 'B-* = Beginning of entity',
        ha='left', va='center', fontsize=10)

# I- tags
i_box = FancyBboxPatch((6.5, legend_y - 0.25), 0.5, 0.5,
                       boxstyle="round,pad=0.05",
                       facecolor=colors['ORG'], edgecolor='black',
                       linewidth=1.5, alpha=0.8)
ax.add_patch(i_box)
ax.text(7.3, legend_y, 'I-* = Inside/continuation of entity',
        ha='left', va='center', fontsize=10)

# O tags
o_box = FancyBboxPatch((11, legend_y - 0.25), 0.5, 0.5,
                       boxstyle="round,pad=0.05",
                       facecolor=colors['O'], edgecolor='black',
                       linewidth=1.5, alpha=0.3)
ax.add_patch(o_box)
ax.text(11.8, legend_y, 'O = Outside any entity',
        ha='left', va='center', fontsize=10)

# Add entity type annotations with arrows
# Person entity
person_arrow = FancyArrowPatch((1.3, 4.5), (1.3, 5.0),
                              arrowstyle='<->', mutation_scale=15,
                              color=colors['PER'], linewidth=2.5, alpha=0.8)
ax.add_patch(person_arrow)
ax.text(1.3, 5.4, 'PERSON', ha='center', va='bottom',
        fontsize=9, fontweight='bold', color=colors['PER'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['PER'], alpha=0.2))

# Organization entity
org_box = FancyBboxPatch((8.5, 4.8), 1.6, 0.5,
                        boxstyle="round,pad=0.05",
                        facecolor=colors['ORG'], edgecolor=colors['ORG'],
                        linewidth=2, alpha=0.2)
ax.add_patch(org_box)
ax.text(9.3, 5.05, 'ORG', ha='center', va='center',
        fontsize=9, fontweight='bold', color=colors['ORG'])

# Location entity
loc_box = FancyBboxPatch((12.5, 4.8), 1.6, 0.5,
                        boxstyle="round,pad=0.05",
                        facecolor=colors['LOC'], edgecolor=colors['LOC'],
                        linewidth=2, alpha=0.2)
ax.add_patch(loc_box)
ax.text(13.3, 5.05, 'LOC', ha='center', va='center',
        fontsize=9, fontweight='bold', color=colors['LOC'])

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-06-nlp/ch28-modern-nlp/diagrams/bio-tagging-scheme.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: bio-tagging-scheme.png")
