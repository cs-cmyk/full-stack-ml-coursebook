"""
Generate Three-Stage Git Workflow Diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Define colors from the palette
color_blue = '#2196F3'
color_green = '#4CAF50'
color_orange = '#FF9800'
color_red = '#F44336'
color_gray = '#607D8B'

# Box dimensions
box_width = 7
box_height = 2
x_center = 5

# Y positions for each stage
y_working = 11
y_staging = 8
y_repo = 5
y_remote = 2

# Helper function to create a stage box
def create_stage_box(y, title, content_lines, color):
    # Main box
    box = FancyBboxPatch(
        (x_center - box_width/2, y - box_height/2),
        box_width, box_height,
        boxstyle="round,pad=0.1",
        linewidth=2.5,
        edgecolor=color,
        facecolor='white',
        zorder=2
    )
    ax.add_patch(box)

    # Title
    ax.text(x_center, y + 0.6, title,
            fontsize=16, fontweight='bold', ha='center', va='center',
            color=color)

    # Content
    y_offset = 0.1
    for line in content_lines:
        ax.text(x_center, y + y_offset, line,
                fontsize=13, ha='center', va='center',
                family='monospace')
        y_offset -= 0.45

# Helper function to create an arrow with label
def create_arrow(y_from, y_to, label, x_offset=0):
    arrow = FancyArrowPatch(
        (x_center + x_offset, y_from - box_height/2 - 0.2),
        (x_center + x_offset, y_to + box_height/2 + 0.2),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        linewidth=2.5,
        color=color_gray,
        zorder=1
    )
    ax.add_patch(arrow)

    # Label background
    y_label = (y_from + y_to) / 2
    label_bg = FancyBboxPatch(
        (x_center + x_offset - 1.2, y_label - 0.25),
        2.4, 0.5,
        boxstyle="round,pad=0.05",
        linewidth=0,
        facecolor='white',
        zorder=3
    )
    ax.add_patch(label_bg)

    # Label text
    ax.text(x_center + x_offset, y_label, label,
            fontsize=12, ha='center', va='center',
            fontweight='bold', color=color_gray,
            zorder=4,
            family='monospace')

# Create the four stages
create_stage_box(y_working, 'Working Directory',
                 ['Your current files (modified)', '• model.py *', '• data.csv'],
                 color_orange)

create_stage_box(y_staging, 'Staging Area',
                 ['Changes marked for commit', '• model.py ✓'],
                 color_blue)

create_stage_box(y_repo, 'Repository',
                 ['Permanent history', '• Commit abc123', '• Commit def456', '• Commit 789xyz'],
                 color_green)

create_stage_box(y_remote, 'Remote (GitHub)',
                 ['Cloud backup + collaboration', '• origin/main'],
                 color_purple := '#9C27B0')

# Create arrows with labels
create_arrow(y_working, y_staging, 'git add')
create_arrow(y_staging, y_repo, 'git commit -m')
create_arrow(y_repo, y_remote, 'git push')

# Add title
ax.text(5, 13.3, 'Git Three-Stage Workflow',
        fontsize=20, fontweight='bold', ha='center', va='center')

# Add caption at bottom
caption_text = ("Git's three-stage workflow: modify files in your working directory, stage logical changes,\n"
                "commit snapshots to permanent history, and push to remote for backup and collaboration.")
ax.text(5, 0.3, caption_text,
        fontsize=11, ha='center', va='center',
        style='italic', wrap=True, color=color_gray)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-02-programming/ch09-version-control/diagrams/git_workflow.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Git workflow diagram saved")
plt.close()
