import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bi-Encoder (Left Panel)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Bi-Encoder Architecture', fontsize=14, fontweight='bold')

# Query path
query_box = FancyBboxPatch((0.5, 7), 2, 1, boxstyle="round,pad=0.1",
                           edgecolor='#2196F3', facecolor='lightblue', linewidth=2)
ax1.add_patch(query_box)
ax1.text(1.5, 7.5, 'Query', ha='center', va='center', fontsize=11, fontweight='bold')

encoder_q = FancyBboxPatch((0.5, 5), 2, 1, boxstyle="round,pad=0.1",
                           edgecolor='#2196F3', facecolor='skyblue', linewidth=2)
ax1.add_patch(encoder_q)
ax1.text(1.5, 5.5, 'Encoder_Q', ha='center', va='center', fontsize=10)

arrow1 = FancyArrowPatch((1.5, 7), (1.5, 6), arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='#2196F3')
ax1.add_patch(arrow1)

q_vec = FancyBboxPatch((0.5, 3), 2, 1, boxstyle="round,pad=0.1",
                       edgecolor='#2196F3', facecolor='lightcyan', linewidth=2)
ax1.add_patch(q_vec)
ax1.text(1.5, 3.5, 'q ∈ ℝᵈ', ha='center', va='center', fontsize=10)

arrow2 = FancyArrowPatch((1.5, 5), (1.5, 4), arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='#2196F3')
ax1.add_patch(arrow2)

# Document path
doc_box = FancyBboxPatch((5.5, 7), 2, 1, boxstyle="round,pad=0.1",
                        edgecolor='#4CAF50', facecolor='lightgreen', linewidth=2)
ax1.add_patch(doc_box)
ax1.text(6.5, 7.5, 'Doc_i', ha='center', va='center', fontsize=11, fontweight='bold')

encoder_d = FancyBboxPatch((5.5, 5), 2, 1, boxstyle="round,pad=0.1",
                          edgecolor='#4CAF50', facecolor='palegreen', linewidth=2)
ax1.add_patch(encoder_d)
ax1.text(6.5, 5.5, 'Encoder_D', ha='center', va='center', fontsize=10)

arrow3 = FancyArrowPatch((6.5, 7), (6.5, 6), arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='#4CAF50')
ax1.add_patch(arrow3)

d_vec = FancyBboxPatch((5.5, 3), 2, 1, boxstyle="round,pad=0.1",
                      edgecolor='#4CAF50', facecolor='honeydew', linewidth=2)
ax1.add_patch(d_vec)
ax1.text(6.5, 3.5, 'd_i ∈ ℝᵈ', ha='center', va='center', fontsize=10)

arrow4 = FancyArrowPatch((6.5, 5), (6.5, 4), arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='#4CAF50')
ax1.add_patch(arrow4)

# Similarity computation
sim_box = FancyBboxPatch((3, 1), 4, 1, boxstyle="round,pad=0.1",
                        edgecolor='#F44336', facecolor='lightyellow', linewidth=2)
ax1.add_patch(sim_box)
ax1.text(5, 1.5, 'cos(q, d_i) = q·d_i / (||q|| ||d_i||)', ha='center', va='center', fontsize=9)

arrow5 = FancyArrowPatch((2.5, 3.5), (4, 2), arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='#2196F3')
ax1.add_patch(arrow5)

arrow6 = FancyArrowPatch((5.5, 3.5), (6, 2), arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='#4CAF50')
ax1.add_patch(arrow6)

ax1.text(5, 0.3, '✓ Pre-compute doc embeddings | O(1) per query',
         ha='center', fontsize=9, style='italic', color='darkgreen')

# Cross-Encoder (Right Panel)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Cross-Encoder Architecture', fontsize=14, fontweight='bold')

# Combined input
combined_box = FancyBboxPatch((2, 7), 6, 1, boxstyle="round,pad=0.1",
                             edgecolor='#9C27B0', facecolor='lavender', linewidth=2)
ax2.add_patch(combined_box)
ax2.text(5, 7.5, '[CLS] query [SEP] document [SEP]', ha='center', va='center',
         fontsize=10, fontweight='bold')

# Single encoder
encoder_cross = FancyBboxPatch((2.5, 4.5), 5, 2, boxstyle="round,pad=0.1",
                              edgecolor='#9C27B0', facecolor='thistle', linewidth=2)
ax2.add_patch(encoder_cross)
ax2.text(5, 5.5, 'Joint Encoder\n(BERT with\ncross-attention)', ha='center', va='center',
         fontsize=10)

arrow7 = FancyArrowPatch((5, 7), (5, 6.5), arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='#9C27B0')
ax2.add_patch(arrow7)

# Relevance score
score_box = FancyBboxPatch((3, 2), 4, 1, boxstyle="round,pad=0.1",
                          edgecolor='#F44336', facecolor='mistyrose', linewidth=2)
ax2.add_patch(score_box)
ax2.text(5, 2.5, 'relevance score ∈ [0,1]', ha='center', va='center', fontsize=10)

arrow8 = FancyArrowPatch((5, 4.5), (5, 3), arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='#9C27B0')
ax2.add_patch(arrow8)

ax2.text(5, 0.3, '✗ Cannot pre-compute | O(N) per query',
         ha='center', fontsize=9, style='italic', color='darkred')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-17/ch49/diagrams/bi_vs_cross_encoder.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Diagram 1 saved: bi_vs_cross_encoder.png")
