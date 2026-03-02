import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Multi-Stage Retrieval Pipeline (Funnel Architecture)',
             fontsize=15, fontweight='bold')

# Stage 0: Full Corpus
stage0 = FancyBboxPatch((1, 8), 10, 1, boxstyle="round,pad=0.1",
                       edgecolor='black', facecolor='lightgray', linewidth=2)
ax.add_patch(stage0)
ax.text(6, 8.5, 'Full Corpus: 1,000,000 documents', ha='center', va='center',
        fontsize=11, fontweight='bold')

# Arrow to Stage 1
arrow1 = FancyArrowPatch((6, 8), (6, 7), arrowstyle='->', mutation_scale=30,
                        linewidth=3, color='#2196F3')
ax.add_patch(arrow1)

# Stage 1: BM25
stage1 = FancyBboxPatch((1.5, 5.5), 9, 1.2, boxstyle="round,pad=0.1",
                       edgecolor='#2196F3', facecolor='lightblue', linewidth=2)
ax.add_patch(stage1)
ax.text(6, 6.3, 'Stage 1: BM25 Sparse Retrieval → 1,000 candidates', ha='center',
        va='center', fontsize=10, fontweight='bold')
ax.text(6, 5.8, 'Latency: 10ms | Recall@1000: 85%', ha='center', va='center',
        fontsize=9, style='italic')

# Arrow to Stage 2
arrow2 = FancyArrowPatch((6, 5.5), (6, 4.5), arrowstyle='->', mutation_scale=30,
                        linewidth=3, color='#4CAF50')
ax.add_patch(arrow2)

# Stage 2: Bi-Encoder
stage2 = FancyBboxPatch((2, 3), 8, 1.2, boxstyle="round,pad=0.1",
                       edgecolor='#4CAF50', facecolor='lightgreen', linewidth=2)
ax.add_patch(stage2)
ax.text(6, 3.8, 'Stage 2: Bi-Encoder Dense → 100 candidates', ha='center',
        va='center', fontsize=10, fontweight='bold')
ax.text(6, 3.3, 'Latency: +40ms (50ms total) | Recall@100: 75%', ha='center',
        va='center', fontsize=9, style='italic')

# Arrow to Stage 3
arrow3 = FancyArrowPatch((6, 3), (6, 2), arrowstyle='->', mutation_scale=30,
                        linewidth=3, color='#F44336')
ax.add_patch(arrow3)

# Stage 3: Cross-Encoder
stage3 = FancyBboxPatch((2.5, 0.5), 7, 1.2, boxstyle="round,pad=0.1",
                       edgecolor='#F44336', facecolor='mistyrose', linewidth=2)
ax.add_patch(stage3)
ax.text(6, 1.3, 'Stage 3: Cross-Encoder Re-rank → 10 final results', ha='center',
        va='center', fontsize=10, fontweight='bold')
ax.text(6, 0.8, 'Latency: +200ms (250ms total) | nDCG@10: 0.89', ha='center',
        va='center', fontsize=9, style='italic')

# Annotation
ax.text(11, 5, 'Each stage\ntrades compute\nfor accuracy', ha='center', va='center',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-17/ch49/diagrams/multi_stage_pipeline.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Diagram 2 saved: multi_stage_pipeline.png")
