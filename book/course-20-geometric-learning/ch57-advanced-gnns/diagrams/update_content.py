#!/usr/bin/env python3
"""Update content.md to add diagram image references after matplotlib code blocks"""

with open('../content.md', 'r') as f:
    lines = f.readlines()

output_lines = []
i = 0
while i < len(lines):
    output_lines.append(lines[i])

    # Check for savefig lines and add image references
    if 'plt.savefig(' in lines[i]:
        # Extract the filename
        if 'architecture_performance_comparison.png' in lines[i]:
            # Look ahead for the comment section that explains it
            # Add image reference after the code block ends
            j = i + 1
            while j < len(lines) and not lines[j].startswith('```'):
                output_lines.append(lines[j])
                j += 1
            if j < len(lines):
                output_lines.append(lines[j])  # Add the closing ```
                output_lines.append('\n')
                output_lines.append('![GNN Architecture Performance](diagrams/architecture_performance_comparison.png)\n')
                output_lines.append('\n')
                i = j

        elif 'heterogeneous_graph_example.png' in lines[i]:
            j = i + 1
            while j < len(lines) and not lines[j].startswith('```'):
                output_lines.append(lines[j])
                j += 1
            if j < len(lines):
                output_lines.append(lines[j])  # Add the closing ```
                output_lines.append('\n')
                output_lines.append('![Heterogeneous Graph Example](diagrams/heterogeneous_graph_example.png)\n')
                output_lines.append('\n')
                i = j

        elif 'sampling_strategy_comparison.png' in lines[i]:
            j = i + 1
            while j < len(lines) and not lines[j].startswith('```'):
                output_lines.append(lines[j])
                j += 1
            if j < len(lines):
                output_lines.append(lines[j])  # Add the closing ```
                output_lines.append('\n')
                output_lines.append('![Sampling Strategy Comparison](diagrams/sampling_strategy_comparison.png)\n')
                output_lines.append('\n')
                i = j

        elif 'sampling_tradeoffs.png' in lines[i]:
            j = i + 1
            while j < len(lines) and not lines[j].startswith('```'):
                output_lines.append(lines[j])
                j += 1
            if j < len(lines):
                output_lines.append(lines[j])  # Add the closing ```
                output_lines.append('\n')
                output_lines.append('![Sampling Trade-offs](diagrams/sampling_tradeoffs.png)\n')
                output_lines.append('\n')
                i = j

    i += 1

# Write back to content.md
with open('../content.md', 'w') as f:
    f.writelines(output_lines)

print("Updated content.md with diagram references")
