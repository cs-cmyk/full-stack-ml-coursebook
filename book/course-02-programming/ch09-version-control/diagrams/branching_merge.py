"""
Generate Branching and Merging Visualization
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.lines as mlines

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.axis('off')

# Define colors
color_blue = '#2196F3'
color_green = '#4CAF50'
color_orange = '#FF9800'
color_purple = '#9C27B0'
color_gray = '#607D8B'

# Commit positions (x, y)
commits = {
    'C1': (2, 4),
    'C2': (4, 4),
    'C3': (6, 4),
    'C4': (5, 2.5),
    'C5': (7, 2.5),
    'C6': (9, 4)
}

# Commit descriptions
descriptions = {
    'C1': 'Initial commit',
    'C2': 'Add baseline model',
    'C3': '(main)',
    'C4': 'Experiment with\nRandom Forest',
    'C5': 'Tune parameters',
    'C6': 'Merge experiment-rf\ninto main'
}

# Draw commit circles
def draw_commit(pos, label, color, is_branch_point=False):
    circle = Circle(pos, 0.35, color=color, ec='white', linewidth=2.5, zorder=3)
    ax.add_patch(circle)
    ax.text(pos[0], pos[1], label,
            fontsize=13, fontweight='bold', ha='center', va='center',
            color='white', zorder=4)

# Draw arrows between commits
def draw_arrow(from_pos, to_pos, color=color_gray):
    arrow = FancyArrowPatch(
        from_pos, to_pos,
        arrowstyle='->,head_width=0.3,head_length=0.3',
        linewidth=2.5,
        color=color,
        zorder=1,
        connectionstyle="arc3,rad=0"
    )
    ax.add_patch(arrow)

# Draw commit descriptions
def draw_description(pos, text, offset_y=-0.9):
    ax.text(pos[0], pos[1] + offset_y, text,
            fontsize=10, ha='center', va='top',
            style='italic', color=color_gray)

# Draw main branch commits
draw_commit(commits['C1'], 'C1', color_blue)
draw_commit(commits['C2'], 'C2', color_blue)
draw_commit(commits['C3'], 'C3', color_blue)
draw_commit(commits['C6'], 'C6', color_green)

# Draw experiment branch commits
draw_commit(commits['C4'], 'C4', color_orange)
draw_commit(commits['C5'], 'C5', color_orange)

# Draw arrows for main branch
draw_arrow((commits['C1'][0] + 0.35, commits['C1'][1]),
          (commits['C2'][0] - 0.35, commits['C2'][1]),
          color_blue)
draw_arrow((commits['C2'][0] + 0.35, commits['C2'][1]),
          (commits['C3'][0] - 0.35, commits['C3'][1]),
          color_blue)

# Draw arrows for experiment branch (diverging)
draw_arrow((commits['C2'][0] + 0.25, commits['C2'][1] - 0.25),
          (commits['C4'][0] - 0.25, commits['C4'][1] + 0.25),
          color_orange)
draw_arrow((commits['C4'][0] + 0.35, commits['C4'][1]),
          (commits['C5'][0] - 0.35, commits['C5'][1]),
          color_orange)

# Draw merge arrows
draw_arrow((commits['C3'][0] + 0.35, commits['C3'][1]),
          (commits['C6'][0] - 0.35, commits['C6'][1]),
          color_green)
draw_arrow((commits['C5'][0] + 0.25, commits['C5'][1] + 0.25),
          (commits['C6'][0] - 0.25, commits['C6'][1] - 0.25),
          color_green)

# Draw descriptions
draw_description(commits['C1'], descriptions['C1'])
draw_description(commits['C2'], descriptions['C2'])
draw_description(commits['C4'], descriptions['C4'], -1.1)
draw_description(commits['C5'], descriptions['C5'])
draw_description(commits['C6'], descriptions['C6'], -1.1)

# Draw branch labels
# Main branch label
main_label = FancyBboxPatch(
    (commits['C3'][0] - 0.6, commits['C3'][1] + 0.6),
    1.2, 0.5,
    boxstyle="round,pad=0.1",
    linewidth=2,
    edgecolor=color_blue,
    facecolor=color_blue,
    zorder=5
)
ax.add_patch(main_label)
ax.text(commits['C3'][0], commits['C3'][1] + 0.85, 'main',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color='white', zorder=6)

# Experiment branch label
exp_label = FancyBboxPatch(
    (commits['C5'][0] - 1.3, commits['C5'][1] + 0.6),
    2.6, 0.5,
    boxstyle="round,pad=0.1",
    linewidth=2,
    edgecolor=color_orange,
    facecolor=color_orange,
    zorder=5
)
ax.add_patch(exp_label)
ax.text(commits['C5'][0], commits['C5'][1] + 0.85, 'experiment-rf',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color='white', zorder=6)

# Add title
ax.text(6.5, 7.3, 'Git Branching and Merging',
        fontsize=20, fontweight='bold', ha='center', va='center')

# Add legend
legend_y = 6.5
ax.text(1.5, legend_y, 'Legend:', fontsize=13, fontweight='bold', color=color_gray)
legend_items = [
    (color_blue, 'Main branch commits'),
    (color_orange, 'Experiment branch commits'),
    (color_green, 'Merge commit')
]
legend_y -= 0.5
for color, label in legend_items:
    circle = Circle((1.8, legend_y), 0.15, color=color, ec='white', linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(2.2, legend_y, label, fontsize=11, va='center', color=color_gray)
    legend_y -= 0.4

# Add caption
caption_text = ("Branches enable parallel development. The experiment branch diverges from main at C2, develops independently (C4, C5),\n"
                "then merges back (C6). Main remains stable throughout experimentation.")
ax.text(6.5, 0.5, caption_text,
        fontsize=11, ha='center', va='center',
        style='italic', color=color_gray)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-02-programming/ch09-version-control/diagrams/branching_merge.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Branching and merging diagram saved")
plt.close()
