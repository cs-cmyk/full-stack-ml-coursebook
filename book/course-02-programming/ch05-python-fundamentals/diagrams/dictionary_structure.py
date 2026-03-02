"""
Dictionary Structure Visualization
Shows key-value pairs and fast lookup
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Define colors
color_blue = '#2196F3'
color_green = '#4CAF50'
color_orange = '#FF9800'
color_purple = '#9C27B0'
color_gray = '#607D8B'

# Title
ax.text(6, 7.5, 'Python Dictionary: Key-Value Pairs', ha='center', va='center',
        fontsize=16, weight='bold')

# Dictionary representation
dict_code = """dataset = {
    "name": "Iris",
    "n_samples": 150,
    "n_features": 4,
    "learning_rate": 0.01
}"""
ax.text(1.5, 5.5, dict_code, ha='left', va='center',
        fontsize=11, family='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))

# Helper function to draw key-value pair
def draw_kv_pair(ax, x, y, key, value, key_color, val_color):
    # Key box
    key_width = 2
    key_height = 0.7
    key_rect = patches.FancyBboxPatch((x, y-key_height/2), key_width, key_height,
                                       boxstyle="round,pad=0.08",
                                       edgecolor=key_color,
                                       facecolor=key_color,
                                       linewidth=2.5,
                                       alpha=0.2)
    ax.add_patch(key_rect)
    ax.text(x + key_width/2, y, key, ha='center', va='center',
            fontsize=11, weight='bold', color=key_color)

    # Arrow
    arrow_x = x + key_width + 0.3
    ax.annotate('', xy=(arrow_x + 0.6, y), xytext=(arrow_x, y),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=color_gray))

    # Value box
    val_x = arrow_x + 1
    val_width = 1.8
    val_height = 0.7
    val_rect = patches.FancyBboxPatch((val_x, y-val_height/2), val_width, val_height,
                                       boxstyle="round,pad=0.08",
                                       edgecolor=val_color,
                                       facecolor=val_color,
                                       linewidth=2.5,
                                       alpha=0.3)
    ax.add_patch(val_rect)
    ax.text(val_x + val_width/2, y, value, ha='center', va='center',
            fontsize=11, weight='bold', color='black')

# Visual representation section
ax.text(7.5, 6.8, 'Visual Representation:', ha='left', va='center',
        fontsize=12, weight='bold', color='black')

# Draw key-value pairs
pairs = [
    ('"name"', '"Iris"', color_blue, color_blue),
    ('"n_samples"', '150', color_green, color_green),
    ('"n_features"', '4', color_orange, color_orange),
    ('"learning_rate"', '0.01', color_purple, color_purple),
]

y_start = 5.8
y_step = 1.0
for i, (key, val, key_col, val_col) in enumerate(pairs):
    draw_kv_pair(ax, 5.5, y_start - i*y_step, key, val, key_col, val_col)

# Labels
ax.text(6.5, 6.5, 'Key', ha='center', va='center',
        fontsize=10, weight='bold', color=color_gray)
ax.text(9.5, 6.5, 'Value', ha='center', va='center',
        fontsize=10, weight='bold', color=color_gray)

# Access examples at bottom
ax.text(6, 1.8, 'Fast O(1) Lookup by Key:', ha='center', va='center',
        fontsize=12, weight='bold', color='black')

examples = [
    ('dataset["name"]', ' → "Iris"', color_blue),
    ('dataset["n_samples"]', ' → 150', color_green),
    ('dataset.get("n_features", 0)', ' → 4', color_orange),
]

y_ex = 1.2
for i, (code, result, col) in enumerate(examples):
    ax.text(2 + i*3.3, y_ex, code, ha='left', va='center',
            fontsize=10, family='monospace', weight='bold', color=col)
    ax.text(2 + i*3.3, y_ex-0.4, result, ha='left', va='center',
            fontsize=10, family='monospace', color=col)

# Key characteristics
chars = [
    '• Keys must be unique',
    '• Keys must be immutable (str, int, tuple)',
    '• Values can be any type',
    '• Unordered (Python 3.7+ maintains insertion order)'
]
y_char = 0.3
for i, char in enumerate(chars):
    ax.text(1.5 + i*2.7, y_char, char, ha='left', va='center',
            fontsize=9, color=color_gray)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-02-programming/ch05-python-fundamentals/diagrams/dictionary_structure.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: dictionary_structure.png")
