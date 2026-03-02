"""
Generate Safe RL / Constrained MDP visualization
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import matplotlib.patches as mpatches

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ============= Left: Constrained vs Unconstrained Policies =============
ax1.set_xlim(-0.5, 6.5)
ax1.set_ylim(-0.5, 6.5)
ax1.set_aspect('equal')
ax1.set_title('Safe RL: Balancing Reward and Safety', fontsize=14, fontweight='bold')
ax1.set_xlabel('State Space Dimension 1', fontsize=11)
ax1.set_ylabel('State Space Dimension 2', fontsize=11)

# Draw unsafe region
unsafe_region = Rectangle((4, 0), 2.5, 6.5, color='#F44336', alpha=0.15,
                         label='Unsafe Region (High Cost)')
ax1.add_patch(unsafe_region)
ax1.text(5.2, 5.5, 'Unsafe\nRegion', fontsize=11, ha='center', va='center',
        color='#F44336', fontweight='bold')

# Draw safe region
safe_region = Rectangle((-0.5, -0.5), 4.5, 7, color='#4CAF50', alpha=0.1,
                       label='Safe Region')
ax1.add_patch(safe_region)
ax1.text(1, 5.8, 'Safe Region', fontsize=11, ha='center', va='center',
        color='#4CAF50', alpha=0.7)

# Start and goal
start = Circle((1, 1), 0.25, color='#2196F3', edgecolor='#1976D2', linewidth=2)
ax1.add_patch(start)
ax1.text(1, 1, 'S', fontsize=12, ha='center', va='center', color='white',
        fontweight='bold')
ax1.text(1, 0.3, 'Start', fontsize=10, ha='center', va='top')

goal = Circle((5.5, 5.5), 0.25, color='#FF9800', edgecolor='#F57C00', linewidth=2)
ax1.add_patch(goal)
ax1.text(5.5, 5.5, 'G', fontsize=12, ha='center', va='center', color='white',
        fontweight='bold')
ax1.text(5.5, 6.2, 'Goal', fontsize=10, ha='center', va='bottom')

# Unconstrained trajectory (goes through unsafe region)
unconstrained_path = np.array([
    [1, 1], [2, 2], [3, 3], [4.2, 4.2], [5, 5], [5.5, 5.5]
])
ax1.plot(unconstrained_path[:, 0], unconstrained_path[:, 1],
        'o-', color='#F44336', linewidth=2.5, markersize=6,
        label='Unconstrained (Shortest, Unsafe)', alpha=0.8)

# Constrained trajectory (avoids unsafe region)
constrained_path = np.array([
    [1, 1], [1.5, 2], [2, 3], [2.5, 4], [3, 5], [3.5, 5.5], [4.5, 5.8], [5.5, 5.5]
])
ax1.plot(constrained_path[:, 0], constrained_path[:, 1],
        's-', color='#4CAF50', linewidth=2.5, markersize=6,
        label='Constrained (Safe, Longer)', alpha=0.8)

ax1.legend(loc='lower left', fontsize=10)
ax1.grid(True, alpha=0.3)

# ============= Right: Constrained MDP Formulation =============
ax2.axis('off')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

# Title
ax2.text(5, 9.5, 'Constrained MDP (CMDP)', fontsize=14, fontweight='bold',
        ha='center', va='top')

# Objective box
objective_text = (
    "Objective: Maximize Expected Return\n\n"
    "max J(π) = 𝔼[∑ᵗ γᵗ R(sₜ, aₜ)]"
)
objective_box = mpatches.FancyBboxPatch((1, 6.5), 8, 2,
                                       boxstyle="round,pad=0.2",
                                       edgecolor='#2196F3', facecolor='#E3F2FD',
                                       linewidth=2)
ax2.add_patch(objective_box)
ax2.text(5, 7.5, objective_text, fontsize=11, ha='center', va='center',
        family='monospace')

# Subject to box
constraint_text = (
    "Subject to: Safety Constraints\n\n"
    "𝔼[∑ᵗ γᵗ C(sₜ, aₜ)] ≤ d\n\n"
    "where C(s,a) is cost, d is threshold"
)
constraint_box = mpatches.FancyBboxPatch((1, 2.5), 8, 3.5,
                                        boxstyle="round,pad=0.2",
                                        edgecolor='#F44336', facecolor='#FFEBEE',
                                        linewidth=2)
ax2.add_patch(constraint_box)
ax2.text(5, 4.25, constraint_text, fontsize=11, ha='center', va='center',
        family='monospace')

# Solution approach box
solution_text = (
    "Solution: Constrained Policy Optimization (CPO)\n"
    "• Uses Lagrangian relaxation: ℒ(π,λ) = J(π) - λ(Jc(π) - d)\n"
    "• Trust region updates with constraint satisfaction\n"
    "• Guarantees constraint satisfaction in expectation"
)
solution_box = mpatches.FancyBboxPatch((1, 0.2), 8, 2,
                                      boxstyle="round,pad=0.2",
                                      edgecolor='#4CAF50', facecolor='#E8F5E9',
                                      linewidth=2)
ax2.add_patch(solution_box)
ax2.text(5, 1.2, solution_text, fontsize=9.5, ha='center', va='center')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-21/ch61/diagrams/safe_rl_concept.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved Safe RL concept diagram to diagrams/safe_rl_concept.png")
plt.close()
