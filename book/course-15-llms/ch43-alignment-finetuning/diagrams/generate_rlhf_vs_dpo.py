"""Generate RLHF vs DPO comparison diagram"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# RLHF Pipeline
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 12)
ax1.axis('off')
ax1.set_title('RLHF: Two-Stage Process', fontsize=14, fontweight='bold')

# Boxes
pref_box1 = FancyBboxPatch((1, 10), 3, 1, boxstyle="round,pad=0.1",
                           edgecolor='#2196F3', facecolor='#cce5ff', linewidth=2)
ax1.add_patch(pref_box1)
ax1.text(2.5, 10.5, 'Preference\nData', ha='center', va='center', fontsize=10)

reward_box = FancyBboxPatch((1, 7), 3, 1.5, boxstyle="round,pad=0.1",
                            edgecolor='#FF9800', facecolor='#ffe5cc', linewidth=2)
ax1.add_patch(reward_box)
ax1.text(2.5, 7.75, 'Reward Model\nTraining', ha='center', va='center', fontsize=10, fontweight='bold')

ppo_box = FancyBboxPatch((1, 4), 3, 1.5, boxstyle="round,pad=0.1",
                         edgecolor='#F44336', facecolor='#ffcccc', linewidth=2)
ax1.add_patch(ppo_box)
ax1.text(2.5, 4.75, 'PPO\nOptimization', ha='center', va='center', fontsize=10, fontweight='bold')

aligned_box1 = FancyBboxPatch((1, 1), 3, 1.5, boxstyle="round,pad=0.1",
                              edgecolor='#4CAF50', facecolor='#ccffcc', linewidth=2)
ax1.add_patch(aligned_box1)
ax1.text(2.5, 1.75, 'Aligned\nModel', ha='center', va='center', fontsize=10, fontweight='bold')

# Arrows
ax1.arrow(2.5, 9.8, 0, -1.8, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=1.5)
ax1.arrow(2.5, 6.8, 0, -1.8, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=1.5)
ax1.arrow(2.5, 3.8, 0, -1.8, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=1.5)

# Labels
ax1.text(5.5, 7.75, 'Phase 1:\nLearn reward\nfunction', ha='left', va='center', fontsize=9, style='italic')
ax1.text(5.5, 4.75, 'Phase 2:\nMaximize reward\nwith KL penalty', ha='left', va='center', fontsize=9, style='italic')

ax1.text(2.5, 0.2, 'Complex • Unstable • Expensive', ha='center', va='center',
         fontsize=10, color='#F44336', fontweight='bold')

# DPO Pipeline
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 12)
ax2.axis('off')
ax2.set_title('DPO: Direct Optimization', fontsize=14, fontweight='bold')

# Boxes
pref_box2 = FancyBboxPatch((1, 10), 3, 1, boxstyle="round,pad=0.1",
                           edgecolor='#2196F3', facecolor='#cce5ff', linewidth=2)
ax2.add_patch(pref_box2)
ax2.text(2.5, 10.5, 'Preference\nData', ha='center', va='center', fontsize=10)

dpo_box = FancyBboxPatch((1, 5.5), 3, 2.5, boxstyle="round,pad=0.1",
                         edgecolor='#9C27B0', facecolor='#e5ccff', linewidth=2)
ax2.add_patch(dpo_box)
ax2.text(2.5, 6.75, 'Direct Preference\nOptimization\n(Single Stage)', ha='center', va='center',
         fontsize=10, fontweight='bold')

aligned_box2 = FancyBboxPatch((1, 1), 3, 1.5, boxstyle="round,pad=0.1",
                              edgecolor='#4CAF50', facecolor='#ccffcc', linewidth=2)
ax2.add_patch(aligned_box2)
ax2.text(2.5, 1.75, 'Aligned\nModel', ha='center', va='center', fontsize=10, fontweight='bold')

# Arrows
ax2.arrow(2.5, 9.8, 0, -1.8, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=1.5)
ax2.arrow(2.5, 5.3, 0, -3.3, head_width=0.3, head_length=0.2, fc='black', ec='black', linewidth=1.5)

# Cross out the eliminated steps
ax2.plot([0.5, 4.5], [8.5, 8.5], 'r--', linewidth=2, alpha=0.5)
ax2.text(5, 8.5, '✗ No reward model needed', ha='left', va='center',
         fontsize=9, color='#F44336', fontweight='bold')

ax2.text(5.5, 6.75, 'Directly optimize\npolicy from\npreferences', ha='left', va='center',
         fontsize=9, style='italic')

ax2.text(2.5, 0.2, 'Simple • Stable • Efficient', ha='center', va='center',
         fontsize=10, color='#4CAF50', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-15/ch43/diagrams/rlhf_vs_dpo.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ RLHF vs DPO comparison diagram saved")
