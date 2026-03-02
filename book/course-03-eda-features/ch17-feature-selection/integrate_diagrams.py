"""
Integrate diagram references into content.md
Adds image references at appropriate locations in the markdown file
"""

import re

# Read the content file
with open('content.md', 'r') as f:
    lines = f.readlines()

# Define insertion points and diagram markdown
insertions = [
    {
        'after_line': 158,  # After "✓ Visualization saved as 'curse_of_dimensionality.png'"
        'content': '''
![Curse of Dimensionality](diagrams/curse_of_dimensionality.png)
*Figure 1: As dimensions increase from 2 to 200, KNN accuracy drops from 88% to 61% (left), while average pairwise distances grow exponentially (right), demonstrating the curse of dimensionality.*

'''
    },
    {
        'after_line': 354,  # After "✓ Visualization saved as 'filter_methods_comparison.png'"
        'content': '''
![Filter Methods Comparison](diagrams/filter_methods_comparison.png)
*Figure 2: Different filter methods (F-test, Chi-squared, Mutual Information) select overlapping but distinct feature sets. Each method captures different aspects of feature importance.*

'''
    },
    {
        'after_line': 534,  # After "Key Insight: RFE finds minimal feature set..."
        'content': '''
![RFE Analysis](diagrams/rfe_analysis.png)
*Figure 3: Left: RFECV cross-validation curve shows the elbow at 7 features, balancing accuracy and simplicity. Right: Feature ranking from RFE, with green bars indicating selected features.*

'''
    },
    {
        'after_line': 756,  # After "Key Insight: L1 penalty drives some coefficients to EXACTLY zero."
        'content': '''
![Lasso Feature Selection](diagrams/lasso_feature_selection.png)
*Figure 4: Left: Lasso regularization path shows coefficients shrinking to zero as α increases. Right: Feature importance from optimal Lasso model (green=selected, gray=eliminated).*

'''
    },
    {
        'after_line': 989,  # After "Key Insight: Permutation importance is more reliable than Gini."
        'content': '''
![Tree Importance Comparison](diagrams/tree_importance_comparison.png)
*Figure 5: Comparing Gini-based importance (left) with permutation importance (center). The scatter plot (right) shows general agreement but some divergence, with permutation importance being more reliable.*

'''
    },
    {
        'after_line': 1257,  # After "Always use sklearn Pipeline to prevent data leakage."
        'content': '''
![Data Leakage Prevention](diagrams/data_leakage_prevention.png)
*Figure 6: Left (WRONG): Computing feature importance on all data before splitting causes leakage. Right (CORRECT): Using Pipeline ensures feature selection uses only training data, preventing overly optimistic performance estimates.*

'''
    }
]

# Check if diagrams are already integrated
already_integrated = any('![' in line and 'diagrams/' in line for line in lines)

if already_integrated:
    print("⚠️  Diagrams appear to already be integrated in content.md")
    print("   If you want to re-integrate, please remove existing diagram references first.")
else:
    # Insert diagrams in reverse order to maintain line numbers
    for insertion in reversed(insertions):
        insert_at = insertion['after_line']
        lines.insert(insert_at, insertion['content'])

    # Write back to file
    with open('content.md', 'w') as f:
        f.writelines(lines)

    print("✓ Successfully integrated all 6 diagrams into content.md")
    print("  - curse_of_dimensionality.png")
    print("  - filter_methods_comparison.png")
    print("  - rfe_analysis.png")
    print("  - lasso_feature_selection.png")
    print("  - tree_importance_comparison.png")
    print("  - data_leakage_prevention.png")
    print("\nPlease review the file to ensure diagrams are placed correctly.")
