#!/bin/bash

# This script updates the content.md file to replace code blocks with image references

# Work on a copy
cp content.md.backup content_updated.md

# The file is complex, so we'll use Python to do the replacements
python3 << 'PYTHON_SCRIPT'
import re

with open('content_updated.md', 'r') as f:
    content = f.read()

# Define the replacements
replacements = [
    # Message passing visualization
    (
        r'### Message Passing Framework\n\n```python\nimport matplotlib\.pyplot.*?```\n\nThe visualization shows',
        '### Message Passing Framework\n\n![Message Passing Visualization](diagrams/message_passing_visualization.png)\n\nThe visualization shows'
    ),
    # Architecture comparison
    (
        r'### Architecture Comparison Diagram\n\n```python\nimport matplotlib\.pyplot.*?```\n\nThis comparison highlights',
        '### Architecture Comparison Diagram\n\n![Architecture Comparison](diagrams/architecture_comparison.png)\n\nThis comparison highlights'
    ),
]

# Apply replacements
for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write the updated content
with open('content.md', 'w') as f:
    f.write(content)

print("Content updated successfully")
PYTHON_SCRIPT

