"""
Generate all diagrams for Chapter 8: SQL Databases
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Color palette
BLUE = '#2196F3'
GREEN = '#4CAF50'
ORANGE = '#FF9800'
RED = '#F44336'
PURPLE = '#9C27B0'
GRAY = '#607D8B'
LIGHT_GRAY = '#E0E0E0'
DARK_GRAY = '#424242'

# ============================================================================
# Diagram 1: CSV vs Database Comparison
# ============================================================================
def create_csv_vs_database():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Left side: CSV Approach
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('CSV Approach', fontsize=16, fontweight='bold', pad=20)

    # CSV files
    files = ['customers.csv', 'orders.csv', 'products.csv']
    y_pos = 8
    for i, fname in enumerate(files):
        rect = FancyBboxPatch((1, y_pos - i*1.2), 6, 0.8,
                              boxstyle="round,pad=0.1",
                              edgecolor=GRAY, facecolor=LIGHT_GRAY, linewidth=2)
        ax1.add_patch(rect)
        ax1.text(4, y_pos - i*1.2 + 0.4, fname, ha='center', va='center',
                fontsize=12, fontweight='bold')

    # Python code steps
    steps = [
        '1. Load all CSVs',
        '2. df.merge() manually',
        '3. Filter in memory'
    ]
    y_pos = 4
    for i, step in enumerate(steps):
        ax1.text(1, y_pos - i*0.8, step, ha='left', va='center',
                fontsize=11, color=DARK_GRAY)

    # Characteristics
    chars = [
        '📊 Memory: Load everything',
        '🐌 Speed: Slow for large data',
        '⚠️  Multi-user: File conflicts'
    ]
    y_pos = 1.5
    for i, char in enumerate(chars):
        ax1.text(1, y_pos - i*0.6, char, ha='left', va='center',
                fontsize=10, color=RED)

    # Right side: Database Approach
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Database Approach', fontsize=16, fontweight='bold', pad=20)

    # Database container
    db_rect = FancyBboxPatch((1, 2.5), 7, 6.5,
                             boxstyle="round,pad=0.2",
                             edgecolor=BLUE, facecolor='white', linewidth=3)
    ax2.add_patch(db_rect)
    ax2.text(4.5, 8.5, 'DATABASE.db', ha='center', va='center',
            fontsize=14, fontweight='bold', color=BLUE)

    # Tables
    tables = [
        ('customers', 7.2),
        ('orders', 5.5),
        ('products', 3.8)
    ]
    for tname, y in tables:
        rect = FancyBboxPatch((2, y), 5, 1,
                              boxstyle="round,pad=0.1",
                              edgecolor=BLUE, facecolor=LIGHT_GRAY, linewidth=2)
        ax2.add_patch(rect)
        ax2.text(4.5, y + 0.5, tname, ha='center', va='center',
                fontsize=11, fontweight='bold')

        # Connection arrows
        if y > 3.8:
            arrow = FancyArrowPatch((4.5, y), (4.5, y - 0.6),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2, color=BLUE, alpha=0.6)
            ax2.add_patch(arrow)

    # SQL Query label
    ax2.text(4.5, 2.0, 'SQL Query:', ha='center', va='center',
            fontsize=11, fontweight='bold', color=BLUE)
    ax2.text(4.5, 1.5, 'SELECT ... WHERE ...', ha='center', va='center',
            fontsize=10, color=DARK_GRAY, style='italic')

    # Characteristics
    chars_db = [
        '📊 Memory: Load only filtered',
        '⚡ Speed: Fast (indexed)',
        '✅ Multi-user: Safe'
    ]
    y_pos = 0.8
    for i, char in enumerate(chars_db):
        ax2.text(1.5, y_pos - i*0.4, char, ha='left', va='center',
                fontsize=10, color=GREEN)

    plt.tight_layout()
    plt.savefig('/home/chirag/ds-book/book/course-02-programming/ch08-sql-databases/diagrams/csv_vs_database.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: csv_vs_database.png")

# ============================================================================
# Diagram 2: SQL Query Execution Order
# ============================================================================
def create_execution_order():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, 'SQL Query Execution Order', ha='center', va='center',
            fontsize=18, fontweight='bold')

    # Written order (left side)
    ax.text(2, 8.5, 'Written Order', ha='center', va='center',
            fontsize=14, fontweight='bold', color=GRAY)

    written_steps = [
        'SELECT columns',
        'FROM table',
        'WHERE condition',
        'GROUP BY columns',
        'HAVING condition',
        'ORDER BY columns',
        'LIMIT n'
    ]

    y_start = 7.5
    for i, step in enumerate(written_steps):
        y = y_start - i * 0.8
        rect = FancyBboxPatch((0.5, y - 0.3), 3, 0.6,
                              boxstyle="round,pad=0.05",
                              edgecolor=GRAY, facecolor=LIGHT_GRAY, linewidth=2)
        ax.add_patch(rect)
        ax.text(2, y, step, ha='center', va='center',
               fontsize=11, family='monospace')

    # Execution order (right side)
    ax.text(7.5, 8.5, 'Execution Order', ha='center', va='center',
            fontsize=14, fontweight='bold', color=BLUE)

    execution_steps = [
        ('1. FROM table', 'Get data source'),
        ('2. WHERE condition', 'Filter rows'),
        ('3. GROUP BY columns', 'Group data'),
        ('4. HAVING condition', 'Filter groups'),
        ('5. SELECT columns', 'Choose columns'),
        ('6. ORDER BY columns', 'Sort results'),
        ('7. LIMIT n', 'Take first n')
    ]

    colors = [BLUE, ORANGE, GREEN, PURPLE, RED, BLUE, GRAY]

    y_start = 7.5
    for i, (step, desc) in enumerate(execution_steps):
        y = y_start - i * 0.8
        rect = FancyBboxPatch((5.5, y - 0.3), 4, 0.6,
                              boxstyle="round,pad=0.05",
                              edgecolor=colors[i], facecolor='white', linewidth=2.5)
        ax.add_patch(rect)
        ax.text(7.5, y + 0.1, step, ha='center', va='center',
               fontsize=11, family='monospace', fontweight='bold')
        ax.text(7.5, y - 0.15, desc, ha='center', va='center',
               fontsize=8, color=DARK_GRAY, style='italic')

        # Arrow from written to execution
        if i > 0:
            arrow = FancyArrowPatch((7.5, y + 0.4), (7.5, y + 0.7),
                                   arrowstyle='->', mutation_scale=15,
                                   linewidth=1.5, color=colors[i], alpha=0.5)
            ax.add_patch(arrow)

    # Warning box
    warning_box = FancyBboxPatch((0.5, 0.5), 9, 1.2,
                                 boxstyle="round,pad=0.1",
                                 edgecolor=RED, facecolor='#FFF3E0', linewidth=2)
    ax.add_patch(warning_box)
    ax.text(5, 1.4, '⚠️  Understanding this order prevents 90% of SQL errors!',
            ha='center', va='center', fontsize=12, fontweight='bold', color=RED)
    ax.text(5, 0.9, 'Example: You can\'t reference a SELECT alias in WHERE',
            ha='center', va='center', fontsize=10, color=DARK_GRAY)
    ax.text(5, 0.6, 'because WHERE executes before SELECT.',
            ha='center', va='center', fontsize=10, color=DARK_GRAY)

    plt.tight_layout()
    plt.savefig('/home/chirag/ds-book/book/course-02-programming/ch08-sql-databases/diagrams/sql_execution_order.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: sql_execution_order.png")

# ============================================================================
# Diagram 3: JOIN Types Visual Guide
# ============================================================================
def create_join_types():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SQL JOIN Types Visual Guide', fontsize=18, fontweight='bold', y=0.98)

    # Sample data visualization
    guests = ['Alice', 'Bob', 'Carol', 'David']
    meals = {'Alice': 'Chicken', 'Carol': 'Fish', 'David': 'Vegan'}

    join_types = [
        ('INNER JOIN', 'Only matches'),
        ('LEFT JOIN', 'All guests + meals'),
        ('RIGHT JOIN', 'All meals + guests'),
        ('FULL OUTER', 'Everything')
    ]

    for idx, (ax, (join_name, description)) in enumerate(zip(axes.flat, join_types)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(f'{join_name}\n({description})',
                    fontsize=14, fontweight='bold', pad=10)

        # Draw two circles representing tables
        circle1 = plt.Circle((3, 5), 2, color=BLUE, alpha=0.3, label='guests')
        circle2 = plt.Circle((7, 5), 2, color=GREEN, alpha=0.3, label='meals')
        ax.add_patch(circle1)
        ax.add_patch(circle2)

        # Labels
        ax.text(2, 8, 'guests', fontsize=12, fontweight='bold', color=BLUE)
        ax.text(7.5, 8, 'meals', fontsize=12, fontweight='bold', color=GREEN)

        # Highlight different regions based on JOIN type
        if idx == 0:  # INNER JOIN
            # Intersection only
            circle_inner = plt.Circle((5, 5), 1.2, color=PURPLE, alpha=0.6)
            ax.add_patch(circle_inner)
            results = [('Alice', 'Chicken'), ('Carol', 'Fish'), ('David', 'Vegan')]

        elif idx == 1:  # LEFT JOIN
            # All of left circle
            circle_left = plt.Circle((3, 5), 2, color=BLUE, alpha=0.5)
            ax.add_patch(circle_left)
            results = [('Alice', 'Chicken'), ('Bob', 'NULL'),
                      ('Carol', 'Fish'), ('David', 'Vegan')]

        elif idx == 2:  # RIGHT JOIN
            # All of right circle
            circle_right = plt.Circle((7, 5), 2, color=GREEN, alpha=0.5)
            ax.add_patch(circle_right)
            results = [('Alice', 'Chicken'), ('Carol', 'Fish'), ('David', 'Vegan')]

        else:  # FULL OUTER
            # Both circles
            circle_both1 = plt.Circle((3, 5), 2, color=BLUE, alpha=0.3)
            circle_both2 = plt.Circle((7, 5), 2, color=GREEN, alpha=0.3)
            ax.add_patch(circle_both1)
            ax.add_patch(circle_both2)
            results = [('Alice', 'Chicken'), ('Bob', 'NULL'),
                      ('Carol', 'Fish'), ('David', 'Vegan')]

        # Display results
        y_pos = 3.5
        ax.text(5, y_pos + 0.5, 'Result:', fontsize=11, fontweight='bold', ha='center')
        for i, (name, meal) in enumerate(results):
            meal_color = DARK_GRAY if meal != 'NULL' else RED
            ax.text(5, y_pos - i*0.5, f'{name} → {meal}',
                   fontsize=10, ha='center', color=meal_color, family='monospace')

    # Add usage guide at bottom
    fig.text(0.5, 0.02,
            'INNER: Only matched records | LEFT: All from left + matches | ' +
            'RIGHT: All from right + matches | FULL: Everything',
            ha='center', fontsize=11, style='italic', color=DARK_GRAY)

    plt.tight_layout()
    plt.savefig('/home/chirag/ds-book/book/course-02-programming/ch08-sql-databases/diagrams/join_types.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: join_types.png")

# ============================================================================
# Diagram 4: Example GROUP BY Results (from code example)
# ============================================================================
def create_groupby_example():
    # Simulate the data from the chapter example
    regions = ['North', 'South', 'East', 'West', 'Central']
    avg_values = [2.089, 2.124, 2.056, 2.145, 2.078]  # Sample values

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(regions, avg_values, color=BLUE, alpha=0.8, edgecolor=DARK_GRAY, linewidth=1.5)

    # Highlight the highest value
    max_idx = avg_values.index(max(avg_values))
    bars[max_idx].set_color(GREEN)

    ax.set_xlabel('Region', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average House Value (100k $)', fontsize=14, fontweight='bold')
    ax.set_title('Average House Value by Region\n(Extracted via SQL GROUP BY)',
                fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add value labels on bars
    for bar, val in zip(bars, avg_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'${val:.2f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/chirag/ds-book/book/course-02-programming/ch08-sql-databases/diagrams/groupby_example.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: groupby_example.png")

# ============================================================================
# Diagram 5: WHERE vs HAVING Visual
# ============================================================================
def create_where_vs_having():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: WHERE (filters rows before grouping)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('WHERE Clause\n(Filters rows BEFORE grouping)',
                 fontsize=14, fontweight='bold', pad=20, color=ORANGE)

    # Original rows
    rows = ['Row 1 ✓', 'Row 2 ✗', 'Row 3 ✓', 'Row 4 ✗', 'Row 5 ✓']
    y_start = 8
    for i, row in enumerate(rows):
        color = GREEN if '✓' in row else LIGHT_GRAY
        alpha = 1.0 if '✓' in row else 0.3
        rect = FancyBboxPatch((1, y_start - i*0.8), 3, 0.6,
                              boxstyle="round,pad=0.05",
                              edgecolor=color, facecolor=color,
                              linewidth=2, alpha=alpha)
        ax1.add_patch(rect)
        ax1.text(2.5, y_start - i*0.8 + 0.3, row, ha='center', va='center',
                fontsize=11, fontweight='bold')

    # Arrow down
    arrow = FancyArrowPatch((2.5, 4), (2.5, 3),
                           arrowstyle='->', mutation_scale=25,
                           linewidth=3, color=ORANGE)
    ax1.add_patch(arrow)
    ax1.text(5, 3.5, 'WHERE\nfilters', ha='center', va='center',
            fontsize=12, fontweight='bold', color=ORANGE)

    # Filtered rows
    filtered = ['Row 1 ✓', 'Row 3 ✓', 'Row 5 ✓']
    y_start = 2
    for i, row in enumerate(filtered):
        rect = FancyBboxPatch((1, y_start - i*0.6), 3, 0.5,
                              boxstyle="round,pad=0.05",
                              edgecolor=GREEN, facecolor=GREEN, linewidth=2)
        ax1.add_patch(rect)
        ax1.text(2.5, y_start - i*0.6 + 0.25, row, ha='center', va='center',
                fontsize=10, fontweight='bold')

    ax1.text(2.5, 0.3, 'Then GROUP BY operates\non filtered rows',
            ha='center', va='center', fontsize=10, style='italic', color=DARK_GRAY)

    # Right: HAVING (filters groups after grouping)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('HAVING Clause\n(Filters groups AFTER aggregation)',
                 fontsize=14, fontweight='bold', pad=20, color=PURPLE)

    # Groups
    groups = [
        ('Group A\nAvg: 3.2', False),
        ('Group B\nAvg: 4.8', True),
        ('Group C\nAvg: 2.1', False),
        ('Group D\nAvg: 5.5', True)
    ]
    y_start = 7.5
    for i, (group, keep) in enumerate(groups):
        color = GREEN if keep else LIGHT_GRAY
        alpha = 1.0 if keep else 0.3
        rect = FancyBboxPatch((6, y_start - i*1.2), 3, 1,
                              boxstyle="round,pad=0.05",
                              edgecolor=color, facecolor=color,
                              linewidth=2, alpha=alpha)
        ax2.add_patch(rect)
        ax2.text(7.5, y_start - i*1.2 + 0.5, group, ha='center', va='center',
                fontsize=10, fontweight='bold')

    # Arrow down
    arrow2 = FancyArrowPatch((7.5, 2.5), (7.5, 1.5),
                            arrowstyle='->', mutation_scale=25,
                            linewidth=3, color=PURPLE)
    ax2.add_patch(arrow2)
    ax2.text(4.5, 2, 'HAVING\nAvg > 4.0', ha='center', va='center',
            fontsize=12, fontweight='bold', color=PURPLE)

    # Filtered groups
    filtered_groups = ['Group B\nAvg: 4.8', 'Group D\nAvg: 5.5']
    y_start = 1
    for i, group in enumerate(filtered_groups):
        rect = FancyBboxPatch((6, y_start - i*0.7), 3, 0.6,
                              boxstyle="round,pad=0.05",
                              edgecolor=GREEN, facecolor=GREEN, linewidth=2)
        ax2.add_patch(rect)
        ax2.text(7.5, y_start - i*0.7 + 0.3, group, ha='center', va='center',
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/chirag/ds-book/book/course-02-programming/ch08-sql-databases/diagrams/where_vs_having.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: where_vs_having.png")

# ============================================================================
# Main execution
# ============================================================================
if __name__ == '__main__':
    print("Generating diagrams for Chapter 8: SQL Databases...")
    print()

    create_csv_vs_database()
    create_execution_order()
    create_join_types()
    create_groupby_example()
    create_where_vs_having()

    print()
    print("✅ All diagrams generated successfully!")
    print("📁 Location: book/course-02-programming/ch08-sql-databases/diagrams/")
