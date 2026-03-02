#!/usr/bin/env python3
"""
Update content.md to replace Mermaid diagrams with image references
"""

import re

# Read the content file
with open('/home/chirag/ds-book/book/course-18/ch51/content.md', 'r') as f:
    content = f.read()

# Define the replacements
replacements = [
    # Diagram 1
    (
        r'```mermaid\ngraph TD\n    subgraph "Batch Prediction \(Offline\)".*?```',
        '![ML System Design Patterns Comparison](diagrams/diagram1_system_patterns.png)'
    ),
    # Diagram 2
    (
        r'```mermaid\ngraph TB\n    subgraph "Data Sources".*?M --> SV\n```',
        '![Feature Store Architecture](diagrams/diagram2_feature_store.png)'
    ),
    # Diagram 3
    (
        r'```mermaid\ngraph TB\n    subgraph "Clients".*?MS1 --> TRACE\n```',
        '![Model Serving Stack](diagrams/diagram3_model_serving.png)'
    )
]

# Apply replacements
for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write the updated content
with open('/home/chirag/ds-book/book/course-18/ch51/content.md', 'w') as f:
    f.write(content)

print("✓ Updated content.md - replaced 3 Mermaid diagrams with image references")
print("  - Diagram 1: diagram1_system_patterns.png")
print("  - Diagram 2: diagram2_feature_store.png")
print("  - Diagram 3: diagram3_model_serving.png")
