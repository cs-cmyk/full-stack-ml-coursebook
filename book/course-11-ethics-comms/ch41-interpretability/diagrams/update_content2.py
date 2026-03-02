#!/usr/bin/env python3
"""Add remaining image references to content.md."""

import re

# Read the content file
content_file = '../content.md'
with open(content_file, 'r') as f:
    content = f.read()

# Define additional replacements for solution sections
replacements = [
    # 6. After wine_importance_comparison code block in Solution 1
    (
        r"(# SHAP accounts for feature interactions more gracefully, explaining the ranking difference\.\n```)\n\n(Both permutation importance and SHAP identify)",
        r"\1\n\n![Wine Feature Importance Comparison](diagrams/wine_importance_comparison.png)\n\n\2"
    ),

    # 7. After diabetes_pdp_ice code block in Solution 2
    (
        r"(# an interaction between BMI and sex, which the averaged PDP hides\.\n```)\n\n(The ICE plot reveals)",
        r"\1\n\n![Diabetes PDP and ICE Plots](diagrams/diabetes_pdp_ice.png)\n\n\2"
    ),
]

# Apply all replacements
for pattern, replacement in replacements:
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content, count=1)
        print(f"Applied replacement for pattern")
    else:
        print(f"Pattern not found")

# Write updated content
with open(content_file, 'w') as f:
    f.write(content)

print("\nUpdated content.md with remaining image references")
