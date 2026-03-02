"""
Control Flow Decision Tree Diagram
Shows if/elif/else structure as a flowchart
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
color_blue = '#2196F3'
color_green = '#4CAF50'
color_orange = '#FF9800'
color_red = '#F44336'
color_gray = '#607D8B'

# Helper function to draw diamond (decision node)
def draw_diamond(ax, x, y, w, h, text, color):
    diamond = patches.FancyBboxPatch((x-w/2, y-h/2), w, h,
                                      boxstyle="round,pad=0.05",
                                      edgecolor=color,
                                      facecolor=color,
                                      linewidth=2.5,
                                      alpha=0.15)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=11, weight='bold', color='black')

# Helper function to draw rectangle (action node)
def draw_rect(ax, x, y, w, h, text, color):
    rect = patches.FancyBboxPatch((x-w/2, y-h/2), w, h,
                                   boxstyle="round,pad=0.1",
                                   edgecolor=color,
                                   facecolor=color,
                                   linewidth=2,
                                   alpha=0.3)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=12, weight='bold', color='black')

# Start node
draw_rect(ax, 5, 9, 1.5, 0.6, 'Start', color_gray)

# Arrow from start
ax.annotate('', xy=(5, 8.2), xytext=(5, 8.7),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# First decision: score >= 85?
draw_diamond(ax, 5, 7.5, 2.2, 0.8, 'score >= 85?', color_blue)

# Yes arrow to Excellence
ax.annotate('', xy=(2.5, 6.5), xytext=(4, 7.1),
            arrowprops=dict(arrowstyle='->', lw=2, color=color_green))
ax.text(3, 7, 'True', ha='center', va='center',
        fontsize=10, weight='bold', color=color_green)
draw_rect(ax, 2.5, 6, 1.8, 0.6, 'Excellence', color_green)

# No arrow to second decision
ax.annotate('', xy=(7.5, 7.5), xytext=(6.1, 7.5),
            arrowprops=dict(arrowstyle='->', lw=2, color=color_red))
ax.text(7, 7.7, 'False', ha='center', va='center',
        fontsize=10, weight='bold', color=color_red)

# Second decision: score >= 60?
draw_diamond(ax, 8.5, 5.5, 2.2, 0.8, 'score >= 60?', color_orange)

# Arrow from first decision to second
ax.annotate('', xy=(8.5, 6.3), xytext=(7.5, 7),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Yes arrow to Pass
ax.annotate('', xy=(6.5, 4.5), xytext=(7.5, 5.1),
            arrowprops=dict(arrowstyle='->', lw=2, color=color_green))
ax.text(6.8, 5, 'True', ha='center', va='center',
        fontsize=10, weight='bold', color=color_green)
draw_rect(ax, 6.5, 4, 1.4, 0.6, 'Pass', color_green)

# No arrow to Fail
ax.annotate('', xy=(8.5, 4.3), xytext=(8.5, 5.1),
            arrowprops=dict(arrowstyle='->', lw=2, color=color_red))
ax.text(8.8, 4.8, 'False', ha='center', va='center',
        fontsize=10, weight='bold', color=color_red)
draw_rect(ax, 8.5, 3.8, 1.4, 0.6, 'Fail', color_red)

# All paths converge to End
ax.annotate('', xy=(5, 1.8), xytext=(2.5, 5.7),
            arrowprops=dict(arrowstyle='->', lw=1.5, color=color_gray,
                          linestyle='dashed'))
ax.annotate('', xy=(5, 1.8), xytext=(6.5, 3.7),
            arrowprops=dict(arrowstyle='->', lw=1.5, color=color_gray,
                          linestyle='dashed'))
ax.annotate('', xy=(5, 1.8), xytext=(8.5, 3.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color=color_gray,
                          linestyle='dashed'))

# End node
draw_rect(ax, 5, 1.2, 1.5, 0.6, 'End', color_gray)

# Title
ax.text(5, 9.7, 'Control Flow: if/elif/else Decision Tree', ha='center', va='center',
        fontsize=16, weight='bold')

# Add code annotation
code_text = """if score >= 85:
    classification = "Excellence"
elif score >= 60:
    classification = "Pass"
else:
    classification = "Fail"
"""
ax.text(0.8, 3.5, code_text, ha='left', va='center',
        fontsize=10, family='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-02-programming/ch05-python-fundamentals/diagrams/control_flow_tree.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: control_flow_tree.png")
