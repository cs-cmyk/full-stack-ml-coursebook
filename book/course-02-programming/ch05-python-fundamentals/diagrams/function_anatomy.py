"""
Function Anatomy Visualization
Shows the components of a Python function
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
color_blue = '#2196F3'
color_green = '#4CAF50'
color_orange = '#FF9800'
color_purple = '#9C27B0'
color_red = '#F44336'
color_gray = '#607D8B'

# Title
ax.text(6, 9.5, 'Python Function Anatomy', ha='center', va='center',
        fontsize=16, weight='bold')

# Function code
function_code = '''def calculate_mean(data, round_digits=2):
    """
    Calculate the arithmetic mean of a list.

    Parameters:
    -----------
    data : list of float
        Numeric values to average
    round_digits : int, optional (default=2)
        Decimal places to round result

    Returns:
    --------
    float : The mean value
    """
    if not data:
        return 0.0

    total = sum(data)
    mean = total / len(data)
    return round(mean, round_digits)'''

# Draw code with background
code_x = 1
code_y = 6.8
ax.text(code_x, code_y, function_code, ha='left', va='top',
        fontsize=9.5, family='monospace',
        bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.95,
                 edgecolor=color_gray, linewidth=2))

# Annotations with arrows pointing to code parts
annotations = [
    # (x, y, text, color, arrow_to_y)
    (9.5, 8.5, 'def keyword\ndeclares function', color_blue, 8.3),
    (9.5, 7.8, 'Function name\n(descriptive, snake_case)', color_green, 8.3),
    (9.5, 7.1, 'Parameters\n(inputs)', color_orange, 8.3),
    (9.5, 6.5, 'Default parameter\n(optional)', color_purple, 8.3),
    (9.5, 5.5, 'Docstring\n(documentation)', color_blue, 7.5),
    (9.5, 4.3, 'Function body\n(indented 4 spaces)', color_green, 5.5),
    (9.5, 3.2, 'Return statement\n(output)', color_red, 4.3),
]

for i, (x, y, text, color, code_y_pos) in enumerate(annotations):
    # Text box
    ax.text(x, y, text, ha='left', va='center',
            fontsize=9, weight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor=color, linewidth=2))

    # Arrow pointing to code
    ax.annotate('', xy=(7.5, code_y_pos), xytext=(x-0.1, y),
                arrowprops=dict(arrowstyle='->', lw=2, color=color,
                              connectionstyle="arc3,rad=0.3"))

# Function call example at bottom
ax.text(6, 1.8, 'Function Call Example:', ha='center', va='center',
        fontsize=12, weight='bold', color='black')

call_example = '''# Calling the function
scores = [85, 92, 78, 95, 88]
average = calculate_mean(scores)
# Result: 87.6

custom = calculate_mean(scores, round_digits=1)
# Result: 87.6'''

ax.text(6, 0.9, call_example, ha='center', va='center',
        fontsize=10, family='monospace',
        bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.9,
                 edgecolor=color_green, linewidth=2))

# Key benefits on the side
benefits = [
    'Benefits:',
    '✓ Reusable code',
    '✓ Clear inputs/outputs',
    '✓ Self-documenting',
    '✓ Easy to test',
    '✓ Maintainable'
]
y_ben = 2.5
for i, benefit in enumerate(benefits):
    weight = 'bold' if i == 0 else 'normal'
    size = 11 if i == 0 else 10
    ax.text(0.5, y_ben - i*0.3, benefit, ha='left', va='center',
            fontsize=size, weight=weight, color=color_gray if i == 0 else 'black')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-02-programming/ch05-python-fundamentals/diagrams/function_anatomy.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: function_anatomy.png")
