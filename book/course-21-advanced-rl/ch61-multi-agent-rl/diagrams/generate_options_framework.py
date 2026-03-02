"""
Generate Options Framework visualization
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Wedge
import numpy as np

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(6, 7.5, 'Hierarchical RL: Options Framework', fontsize=16, fontweight='bold',
        ha='center', va='top')

# ============= High-level Policy (Top) =============
y_high = 5.5

# Agent box
agent_box = FancyBboxPatch((4.5, y_high + 0.5), 3, 0.8,
                          boxstyle="round,pad=0.1",
                          edgecolor='#2196F3', facecolor='#E3F2FD',
                          linewidth=3)
ax.add_patch(agent_box)
ax.text(6, y_high + 0.9, 'Policy over Options μ(ω|s)', fontsize=12,
        ha='center', va='center', fontweight='bold')

ax.text(1, y_high + 0.9, 'High-level:', fontsize=12, fontweight='bold',
        ha='left', va='center', color='#2196F3')

# Current state
state_circle = patches.Circle((6, y_high), 0.35, color='#FF9800',
                             edgecolor='#F57C00', linewidth=2)
ax.add_patch(state_circle)
ax.text(6, y_high, 's', fontsize=14, ha='center', va='center',
        color='white', fontweight='bold')

# Arrows to options
option_positions = [(2, 3), (5, 3), (8, 3), (10.5, 3)]
option_colors = ['#4CAF50', '#9C27B0', '#F44336', '#607D8B']
option_names = ['Option ω₁', 'Option ω₂', 'Option ω₃', 'Option ω₄']

for i, (x, y) in enumerate(option_positions):
    # Arrow from state to option
    arrow = FancyArrowPatch((6, y_high - 0.3), (x, y + 1.2),
                           arrowstyle='->', mutation_scale=15,
                           color=option_colors[i], linewidth=2, alpha=0.6)
    ax.add_patch(arrow)

# ============= Options (Middle) =============
y_option = 3

ax.text(1, y_option + 0.5, 'Options\n(temporally\nextended\nactions):',
        fontsize=11, fontweight='bold', ha='left', va='center',
        color='#607D8B')

for i, (x, y) in enumerate(option_positions):
    if i < 3:  # Only detail first 3 options
        # Option box
        box = FancyBboxPatch((x - 0.8, y), 1.6, 1.2,
                            boxstyle="round,pad=0.1",
                            edgecolor=option_colors[i],
                            facecolor=option_colors[i], alpha=0.2,
                            linewidth=2)
        ax.add_patch(box)

        # Option name
        ax.text(x, y + 1.0, option_names[i], fontsize=11,
               ha='center', va='center', fontweight='bold',
               color=option_colors[i])

        # Components
        ax.text(x, y + 0.7, 'I(s): initiation', fontsize=8,
               ha='center', va='center')
        ax.text(x, y + 0.45, 'π(a|s): policy', fontsize=8,
               ha='center', va='center')
        ax.text(x, y + 0.2, 'β(s): termination', fontsize=8,
               ha='center', va='center')
    else:
        # Ellipsis for additional options
        ax.text(x, y + 0.5, '⋯', fontsize=24, ha='center', va='center',
               color=option_colors[i], fontweight='bold')

# ============= Low-level Execution (Bottom) =============
y_low = 0.5

ax.text(1, y_low + 0.8, 'Low-level\nExecution:', fontsize=11, fontweight='bold',
        ha='left', va='center', color='#FF9800')

# Example trajectory for one option
trajectory_y = y_low + 0.5
trajectory_x_start = 3
trajectory_x_end = 9

# Draw trajectory
n_steps = 7
x_positions = np.linspace(trajectory_x_start, trajectory_x_end, n_steps)
y_positions = trajectory_y + 0.2 * np.sin(np.linspace(0, 2*np.pi, n_steps))

for i in range(n_steps - 1):
    arrow = FancyArrowPatch((x_positions[i], y_positions[i]),
                           (x_positions[i+1], y_positions[i+1]),
                           arrowstyle='->', mutation_scale=12,
                           color='#4CAF50', linewidth=2)
    ax.add_patch(arrow)

    # Draw state
    if i > 0:
        circle = patches.Circle((x_positions[i], y_positions[i]), 0.15,
                               color='#4CAF50', alpha=0.5)
        ax.add_patch(circle)
        ax.text(x_positions[i], y_positions[i] - 0.4, f's{i}',
               fontsize=8, ha='center', va='top')

# Start and end
start_circle = patches.Circle((x_positions[0], y_positions[0]), 0.2,
                             color='#4CAF50', edgecolor='#388E3C',
                             linewidth=2)
ax.add_patch(start_circle)
ax.text(x_positions[0], y_positions[0] - 0.4, 'start',
       fontsize=8, ha='center', va='top', fontweight='bold')

end_circle = patches.Circle((x_positions[-1], y_positions[-1]), 0.2,
                           color='#F44336', edgecolor='#D32F2F',
                           linewidth=2)
ax.add_patch(end_circle)
ax.text(x_positions[-1], y_positions[-1] - 0.4, 'terminate',
       fontsize=8, ha='center', va='top', fontweight='bold')

# Label
ax.text(6, trajectory_y - 1, 'Selected option executes its policy π(a|s) until termination β(s)',
       fontsize=10, ha='center', va='center', style='italic',
       bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))

# Add benefits box
benefits_text = ("Benefits:\n"
                "• Temporal abstraction\n"
                "• Sample efficiency\n"
                "• Transfer learning\n"
                "• Interpretable sub-goals")
benefits_box = ax.text(11.5, 5.5, benefits_text, fontsize=9,
                      ha='right', va='top',
                      bbox=dict(boxstyle='round', facecolor='#FFF9C4',
                               edgecolor='#FF9800', linewidth=2))

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-21/ch61/diagrams/options_framework.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved Options framework diagram to diagrams/options_framework.png")
plt.close()
