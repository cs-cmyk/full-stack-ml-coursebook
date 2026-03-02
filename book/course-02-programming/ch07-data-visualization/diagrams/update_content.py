#!/usr/bin/env python3
"""
Update content.md to include references to generated diagrams.
"""

# Read the content file
with open('../content.md', 'r') as f:
    content = f.read()

# Define replacements to add diagram references
replacements = [
    # 1. Add matplotlib architecture diagram after the architecture description
    (
        '**Best practice:** Use the explicit object-oriented interface (`fig, ax = plt.subplots()`) rather than the pyplot state-based interface for clarity and scalability.',
        '''**Best practice:** Use the explicit object-oriented interface (`fig, ax = plt.subplots()`) rather than the pyplot state-based interface for clarity and scalability.

![Matplotlib Architecture](diagrams/matplotlib_architecture.svg)
*Figure: Matplotlib's object-oriented architecture showing the relationship between Figure, Axes, Axis, and Artists components.*'''
    ),

    # 2. Add chart decision tree and comparison after the visual overview
    (
        '''└─ Change over TIME
   └─ Line plot, Area chart
```

## Code Example: Complete Visualization Workflow''',
        '''└─ Change over TIME
   └─ Line plot, Area chart
```

For a comprehensive comparison of common chart types, see the visual guide below:

![Chart Types Comparison](diagrams/chart_types_comparison.png)
*Figure: Common chart types and their use cases. Each visualization type is optimized for specific data structures and questions.*

## Code Example: Complete Visualization Workflow'''
    ),

    # 3. Add reference to univariate analysis diagram
    (
        '''# Output: 4-panel figure showing distributions with:
# - Right-skewed income distribution (mean > median)
# - Clear outliers visible in box plot
# - House values capped at $500k (visible clustering at max)
# - Smooth KDE reveals distribution shape without binning artifacts''',
        '''# Output: 4-panel figure showing distributions with:
# - Right-skewed income distribution (mean > median)
# - Clear outliers visible in box plot
# - House values capped at $500k (visible clustering at max)
# - Smooth KDE reveals distribution shape without binning artifacts

# See the generated figure:
# ![Univariate Analysis](diagrams/univariate_analysis.png)'''
    ),

    # 4. Add reference to bivariate analysis diagram
    (
        '''# - Hexbin reveals density patterns invisible in standard scatter plot
# - Most data concentrated in 2-5 income range, 1-3 house value range''',
        '''# - Hexbin reveals density patterns invisible in standard scatter plot
# - Most data concentrated in 2-5 income range, 1-3 house value range

# See the generated figure:
# ![Bivariate Analysis](diagrams/bivariate_analysis.png)'''
    ),

    # 5. Add reference to categorical comparison diagram
    (
        '''# - Strip plot shows individual data points with statistical summary
# - 95% confidence intervals overlap, suggesting no dramatic differences''',
        '''# - Strip plot shows individual data points with statistical summary
# - 95% confidence intervals overlap, suggesting no dramatic differences

# See the generated figure:
# ![Categorical Comparison](diagrams/categorical_comparison.png)'''
    ),

    # 6. Add reference to multivariate analysis diagram
    (
        '''# Pair plot reveals:
# - Income vs. value relationship is roughly linear but with spread
# - Some outliers visible in AveRooms (mansions with 20+ rooms)''',
        '''# Pair plot reveals:
# - Income vs. value relationship is roughly linear but with spread
# - Some outliers visible in AveRooms (mansions with 20+ rooms)

# See the generated figure:
# ![Multivariate Analysis](diagrams/multivariate_analysis.png)'''
    ),

    # 7. Add reference to publication quality diagram
    (
        '''# Output: A polished, publication-ready scatter plot showing:
# - Four dimensions of data (x, y, color, size)
# - Clear labels with units
# - Reference line for interpretation
# - Summary statistics in annotation box
# - Professional styling suitable for reports or presentations''',
        '''# Output: A polished, publication-ready scatter plot showing:
# - Four dimensions of data (x, y, color, size)
# - Clear labels with units
# - Reference line for interpretation
# - Summary statistics in annotation box
# - Professional styling suitable for reports or presentations

# See the generated figure:
# ![Publication Quality](diagrams/publication_quality.png)'''
    ),
]

# Apply all replacements
for old_text, new_text in replacements:
    if old_text in content:
        content = content.replace(old_text, new_text)
        print(f"✓ Added diagram reference")
    else:
        print(f"✗ Could not find text to replace")

# Write the updated content
with open('../content.md', 'w') as f:
    f.write(content)

print("\n✓ Content file updated with diagram references!")
