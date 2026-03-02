#!/usr/bin/env python3
"""
Update image references in content.md to point to diagrams/ directory.
"""

import re

content_file = '../content.md'

# Read the content
with open(content_file, 'r') as f:
    content = f.read()

# Define the changes to make
changes = [
    # Change 1: Update existing image reference
    {
        'old': '![Text Preprocessing Pipeline](preprocessing_pipeline.png)',
        'new': '![Text Preprocessing Pipeline](diagrams/preprocessing_pipeline.png)'
    },
    # Change 2: Add image reference after bow_matrix_heatmap code block
    {
        'old': "plt.savefig('bow_matrix_heatmap.png', dpi=150, bbox_inches='tight')\nplt.show()\n\n# Output:",
        'new': "plt.savefig('bow_matrix_heatmap.png', dpi=150, bbox_inches='tight')\nplt.show()\n```\n\n![Bag-of-Words Document-Term Matrix](diagrams/bow_matrix_heatmap.png)\n\n```python\n# Output:"
    },
    # Change 3: Add image reference after bow_vs_tfidf code block
    {
        'old': "plt.savefig('bow_vs_tfidf.png', dpi=150, bbox_inches='tight')\nplt.show()\n\n# Calculate IDF values",
        'new': "plt.savefig('bow_vs_tfidf.png', dpi=150, bbox_inches='tight')\nplt.show()\n```\n\n![BoW vs TF-IDF Comparison](diagrams/bow_vs_tfidf.png)\n\n```python\n# Calculate IDF values"
    },
    # Change 4: Add image reference after ngram_extraction code block
    {
        'old': "plt.savefig('ngram_extraction.png', dpi=150, bbox_inches='tight')\nplt.show()\n\n# Output:",
        'new': "plt.savefig('ngram_extraction.png', dpi=150, bbox_inches='tight')\nplt.show()\n```\n\n![N-gram Extraction Visualization](diagrams/ngram_extraction.png)\n\n```python\n# Output:"
    },
    # Change 5: Add image reference after similarity_comparison code block
    {
        'old': "plt.savefig('similarity_comparison.png', dpi=150)\nplt.show()\n\n# Output:",
        'new': "plt.savefig('similarity_comparison.png', dpi=150)\nplt.show()\n```\n\n![Document Similarity Comparison](diagrams/similarity_comparison.png)\n\n```python\n# Output:"
    },
]

# Apply each change
for i, change in enumerate(changes, 1):
    old_text = change['old']
    new_text = change['new']

    if old_text in content:
        content = content.replace(old_text, new_text, 1)  # Replace only first occurrence
        print(f"✓ Change {i} applied successfully")
    else:
        print(f"✗ Change {i} NOT found - pattern may have changed")
        print(f"  Looking for: {old_text[:50]}...")

# Write the updated content
with open(content_file, 'w') as f:
    f.write(content)

print(f"\n✓ Updated {content_file}")
print("All image references now point to diagrams/ directory")
