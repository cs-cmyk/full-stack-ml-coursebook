#!/usr/bin/env python3
"""Update content.md to add image references after code blocks."""

import re

# Read the content file
content_file = '../content.md'
with open(content_file, 'r') as f:
    content = f.read()

# Define replacements: (search_pattern, replacement)
replacements = [
    # 1. After interpretability_spectrum code block
    (
        r"(# Output:\n# \[A horizontal bar chart showing models from most interpretable \(linear regression\)\n#  to least interpretable \(large transformers\), with color gradient and annotations\]\n```)\n\n(The visualization shows the interpretability spectrum\.)",
        r"\1\n\n![Interpretability Spectrum](diagrams/interpretability_spectrum.png)\n\n\2"
    ),

    # 2. After permutation_importance code block
    (
        r"(#   worst compactness         0\.0121         0\.0008\n```)\n\n(Permutation importance answers:)",
        r"\1\n\n![Permutation Feature Importance](diagrams/permutation_importance.png)\n\n\2"
    ),

    # 3. After shap_waterfall code block
    (
        r"(# Difference: 0\.000000 \(should be ~0\)\n```)\n\n(SHAP \(SHapley Additive exPlanations\))",
        r"\1\n\n![SHAP Waterfall Plot](diagrams/shap_waterfall.png)\n\n\2"
    ),

    # 4. After shap_summary code block
    (
        r"(#   worst compactness         0\.0287\n```)\n\n(The SHAP summary plot)",
        r"\1\n\n![SHAP Summary Plot](diagrams/shap_summary.png)\n\n\2"
    ),

    # 5. After pdp_ice_plot code block
    (
        r"(#    others plateau around MedInc=5, revealing subgroups \(e\.g\., coastal vs\. inland\)\]\n```)\n\n(Partial Dependence Plots)",
        r"\1\n\n![Partial Dependence and ICE Plots](diagrams/pdp_ice_plot.png)\n\n\2"
    ),

    # 6. After wine_importance_comparison code block in Solution 1
    (
        r"(# while SHAP's game-theoretic approach accounts for feature interactions more robustly\.\n```)\n\n(Both permutation importance and SHAP identify)",
        r"\1\n\n![Wine Feature Importance Comparison](diagrams/wine_importance_comparison.png)\n\n\2"
    ),

    # 7. After diabetes_pdp_ice code block in Solution 2
    (
        r"(# an interaction between BMI and sex, which the PDP's averaging obscures\.\n```)\n\n(The ICE plot reveals)",
        r"\1\n\n![Diabetes PDP and ICE Plots](diagrams/diabetes_pdp_ice.png)\n\n\2"
    ),
]

# Apply all replacements
for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content, count=1)

# Write updated content
with open(content_file, 'w') as f:
    f.write(content)

print("Updated content.md with image references")
print("Added 7 image references to the document")
