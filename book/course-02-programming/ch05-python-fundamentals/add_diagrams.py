#!/usr/bin/env python3
"""
Script to add diagram references to content.md at the correct locations.
"""

# Read the content
with open('content.md', 'r') as f:
    lines = f.readlines()

# Define insertions: (after_line_number, text_to_insert)
# Line numbers are 0-indexed in the list
insertions = [
    (85, "\n![Variable Reference Model](diagrams/variable_reference_model.png)\n"),
    (236, "\n![Control Flow Decision Tree](diagrams/control_flow_tree.png)\n"),
    (428, "\n![Zero-Based Indexing](diagrams/zero_based_indexing.png)\n"),
    (451, "\n![List Slicing](diagrams/list_slicing.png)\n"),
    (649, "\n![Dictionary Structure](diagrams/dictionary_structure.png)\n"),
    (913, "\n![Function Anatomy](diagrams/function_anatomy.png)\n"),
]

# Sort in reverse order to maintain line numbers as we insert
insertions.sort(reverse=True)

# Insert diagrams
for line_num, text in insertions:
    lines.insert(line_num + 1, text)

# Write back
with open('content.md', 'w') as f:
    f.writelines(lines)

print("Successfully added 6 diagram references to content.md")
print("Locations:")
for line_num, text in sorted(insertions):
    print(f"  - After line {line_num + 1}: {text.strip()}")
