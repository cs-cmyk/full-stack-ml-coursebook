"""
Generate Inverse RL conceptual diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

# Title
ax.text(6, 5.5, 'Standard RL vs Inverse RL', fontsize=16, fontweight='bold',
        ha='center', va='top')

# ============= Standard RL (Top) =============
y_standard = 4

# Reward function box
reward_box = FancyBboxPatch((0.5, y_standard), 2, 0.8,
                            boxstyle="round,pad=0.1",
                            edgecolor='#2196F3', facecolor='#E3F2FD',
                            linewidth=2)
ax.add_patch(reward_box)
ax.text(1.5, y_standard + 0.4, 'Reward\nFunction R(s,a)', fontsize=11,
        ha='center', va='center', fontweight='bold')

# Arrow to RL algorithm
arrow1 = FancyArrowPatch((2.5, y_standard + 0.4), (4, y_standard + 0.4),
                        arrowstyle='->', mutation_scale=20,
                        color='#607D8B', linewidth=2)
ax.add_patch(arrow1)
ax.text(3.25, y_standard + 0.8, 'given', fontsize=10, ha='center',
        style='italic', color='#607D8B')

# RL Algorithm box
rl_box = FancyBboxPatch((4, y_standard), 2.5, 0.8,
                       boxstyle="round,pad=0.1",
                       edgecolor='#FF9800', facecolor='#FFF3E0',
                       linewidth=2)
ax.add_patch(rl_box)
ax.text(5.25, y_standard + 0.4, 'RL Algorithm\n(Q-learning, PPO, etc.)',
        fontsize=11, ha='center', va='center', fontweight='bold')

# Arrow to policy
arrow2 = FancyArrowPatch((6.5, y_standard + 0.4), (8, y_standard + 0.4),
                        arrowstyle='->', mutation_scale=20,
                        color='#607D8B', linewidth=2)
ax.add_patch(arrow2)
ax.text(7.25, y_standard + 0.8, 'learns', fontsize=10, ha='center',
        style='italic', color='#607D8B')

# Policy box
policy_box = FancyBboxPatch((8, y_standard), 2, 0.8,
                           boxstyle="round,pad=0.1",
                           edgecolor='#4CAF50', facecolor='#E8F5E9',
                           linewidth=2)
ax.add_patch(policy_box)
ax.text(9, y_standard + 0.4, 'Optimal\nPolicy π*(s)', fontsize=11,
        ha='center', va='center', fontweight='bold')

# ============= Inverse RL (Bottom) =============
y_inverse = 1.5

# Expert demonstrations box
demo_box = FancyBboxPatch((0.5, y_inverse), 2, 0.8,
                         boxstyle="round,pad=0.1",
                         edgecolor='#4CAF50', facecolor='#E8F5E9',
                         linewidth=2)
ax.add_patch(demo_box)
ax.text(1.5, y_inverse + 0.4, 'Expert\nDemonstrations', fontsize=11,
        ha='center', va='center', fontweight='bold')

# Arrow to IRL algorithm
arrow3 = FancyArrowPatch((2.5, y_inverse + 0.4), (4, y_inverse + 0.4),
                        arrowstyle='->', mutation_scale=20,
                        color='#607D8B', linewidth=2)
ax.add_patch(arrow3)
ax.text(3.25, y_inverse + 0.8, 'observed', fontsize=10, ha='center',
        style='italic', color='#607D8B')

# IRL Algorithm box
irl_box = FancyBboxPatch((4, y_inverse), 2.5, 0.8,
                        boxstyle="round,pad=0.1",
                        edgecolor='#9C27B0', facecolor='#F3E5F5',
                        linewidth=2)
ax.add_patch(irl_box)
ax.text(5.25, y_inverse + 0.4, 'IRL Algorithm\n(MaxEnt IRL, etc.)',
        fontsize=11, ha='center', va='center', fontweight='bold')

# Arrow to reward
arrow4 = FancyArrowPatch((6.5, y_inverse + 0.4), (8, y_inverse + 0.4),
                        arrowstyle='->', mutation_scale=20,
                        color='#607D8B', linewidth=2)
ax.add_patch(arrow4)
ax.text(7.25, y_inverse + 0.8, 'infers', fontsize=10, ha='center',
        style='italic', color='#607D8B')

# Recovered reward box
recovered_box = FancyBboxPatch((8, y_inverse), 2, 0.8,
                              boxstyle="round,pad=0.1",
                              edgecolor='#2196F3', facecolor='#E3F2FD',
                              linewidth=2)
ax.add_patch(recovered_box)
ax.text(9, y_inverse + 0.4, 'Recovered\nReward R*(s)', fontsize=11,
        ha='center', va='center', fontweight='bold')

# Labels
ax.text(0.2, y_standard + 0.4, 'Forward:', fontsize=12, fontweight='bold',
        ha='right', va='center', color='#FF9800')
ax.text(0.2, y_inverse + 0.4, 'Inverse:', fontsize=12, fontweight='bold',
        ha='right', va='center', color='#9C27B0')

# Add explanation box
explanation = ("IRL reverses the standard RL pipeline:\n"
              "Instead of learning behavior from rewards,\n"
              "it infers what rewards explain observed behavior.")
ax.text(6, 0.3, explanation, fontsize=10, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8),
        style='italic')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-21/ch61/diagrams/irl_concept.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved IRL concept diagram to diagrams/irl_concept.png")
plt.close()
