"""
Variable Reference Model Diagram
Shows how variables are references to objects in memory
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Define colors
color_blue = '#2196F3'
color_green = '#4CAF50'
color_gray = '#607D8B'

# Variable names (labels)
var_y = 4
ax.add_patch(patches.FancyBboxPatch((0.5, var_y-0.3), 1.8, 0.6,
                                     boxstyle="round,pad=0.1",
                                     edgecolor=color_blue,
                                     facecolor='white',
                                     linewidth=2))
ax.text(1.4, var_y, 'n_samples', ha='center', va='center',
        fontsize=12, weight='bold', color=color_blue)

ax.add_patch(patches.FancyBboxPatch((0.5, var_y-1.8), 1.8, 0.6,
                                     boxstyle="round,pad=0.1",
                                     edgecolor=color_green,
                                     facecolor='white',
                                     linewidth=2))
ax.text(1.4, var_y-1.5, 'learning_rate', ha='center', va='center',
        fontsize=12, weight='bold', color=color_green)

# Objects in memory
obj_y = 4
ax.add_patch(patches.FancyBboxPatch((5, obj_y-0.4), 1.5, 0.8,
                                     boxstyle="round,pad=0.1",
                                     edgecolor=color_blue,
                                     facecolor=color_blue,
                                     linewidth=2,
                                     alpha=0.2))
ax.text(5.75, obj_y, '150', ha='center', va='center',
        fontsize=16, weight='bold', color=color_blue)
ax.text(5.75, obj_y-0.6, 'int object', ha='center', va='center',
        fontsize=9, style='italic', color=color_gray)

ax.add_patch(patches.FancyBboxPatch((5, obj_y-1.9), 1.5, 0.8,
                                     boxstyle="round,pad=0.1",
                                     edgecolor=color_green,
                                     facecolor=color_green,
                                     linewidth=2,
                                     alpha=0.2))
ax.text(5.75, obj_y-1.5, '0.01', ha='center', va='center',
        fontsize=16, weight='bold', color=color_green)
ax.text(5.75, obj_y-2.1, 'float object', ha='center', va='center',
        fontsize=9, style='italic', color=color_gray)

# Arrows (references)
ax.annotate('', xy=(5, var_y), xytext=(2.3, var_y),
            arrowprops=dict(arrowstyle='->', lw=2, color=color_blue))
ax.annotate('', xy=(5, var_y-1.5), xytext=(2.3, var_y-1.5),
            arrowprops=dict(arrowstyle='->', lw=2, color=color_green))

# Labels
ax.text(3.6, var_y+0.3, 'references', ha='center', va='center',
        fontsize=10, style='italic', color=color_gray)
ax.text(3.6, var_y-1.2, 'references', ha='center', va='center',
        fontsize=10, style='italic', color=color_gray)

# Section labels
ax.text(1.4, 5.2, 'Variable Names', ha='center', va='center',
        fontsize=13, weight='bold', color='black')
ax.text(5.75, 5.2, 'Objects in Memory', ha='center', va='center',
        fontsize=13, weight='bold', color='black')

# Title
ax.text(5, 5.7, 'Python Variable Reference Model', ha='center', va='center',
        fontsize=16, weight='bold')

# Add explanatory text
ax.text(5, 0.5, 'Variables are labels that reference objects, not containers that hold values',
        ha='center', va='center', fontsize=11, style='italic', color=color_gray)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-02-programming/ch05-python-fundamentals/diagrams/variable_reference_model.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: variable_reference_model.png")
