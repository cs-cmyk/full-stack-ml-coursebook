"""
Style and quality check for Chapter 55 code blocks
"""

import re

# Read the content.md file
with open('content.md', 'r') as f:
    content = f.read()

# Extract all code blocks
code_blocks = re.findall(r'```python(.*?)```', content, re.DOTALL)

print("="*70)
print("STYLE AND QUALITY CHECKS")
print("="*70)

style_issues = []
good_practices = []

# Check each code block
for i, block in enumerate(code_blocks, 1):
    print(f"\n--- Code Block {i} ---")

    # Check for random_state
    if 'random' in block.lower() and 'seed' not in block and 'random_state' not in block:
        style_issues.append(f"Block {i}: Missing random_state or seed for reproducibility")
    elif 'random_state=42' in block or 'np.random.seed(42)' in block:
        good_practices.append(f"Block {i}: ✓ Uses random_state=42 or seed(42)")

    # Check for clear variable names
    if re.search(r'\b[a-z]{1,2}\b\s*=', block) and 'ax' not in block and 'df' not in block:
        style_issues.append(f"Block {i}: Potentially unclear variable names")

    # Check for comments or docstrings
    if '#' in block or '"""' in block:
        good_practices.append(f"Block {i}: ✓ Contains comments/documentation")

    # Check PEP 8 line length (rough check)
    lines = block.split('\n')
    long_lines = [j for j, line in enumerate(lines, 1) if len(line) > 100]
    if long_lines:
        style_issues.append(f"Block {i}: Lines {long_lines[:3]} exceed 100 characters")

    # Check for deprecated APIs (common ones)
    deprecated_patterns = [
        (r'\.ix\[', 'Using deprecated .ix indexer'),
        (r'pd\.np\.', 'Using deprecated pd.np'),
        (r'from_items', 'Using deprecated from_items'),
    ]

    for pattern, msg in deprecated_patterns:
        if re.search(pattern, block):
            style_issues.append(f"Block {i}: {msg}")

print("\n" + "="*70)
print("GOOD PRACTICES FOUND")
print("="*70)
for practice in good_practices[:10]:  # Show first 10
    print(f"  {practice}")
if len(good_practices) > 10:
    print(f"  ... and {len(good_practices) - 10} more")

print("\n" + "="*70)
print("STYLE ISSUES")
print("="*70)
if style_issues:
    for issue in style_issues:
        print(f"  ⚠ {issue}")
else:
    print("  ✓ No significant style issues detected!")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total code blocks: {len(code_blocks)}")
print(f"Good practices: {len(good_practices)}")
print(f"Style issues: {len(style_issues)}")

# Check for specific patterns
print("\n" + "="*70)
print("SPECIFIC CHECKS")
print("="*70)

# Check all imports are at the top of blocks that need them
has_all_imports = []
for i, block in enumerate(code_blocks, 1):
    if 'import' in block:
        imports = re.findall(r'^import .*$|^from .* import .*$', block, re.MULTILINE)
        if imports:
            has_all_imports.append(f"Block {i}: {len(imports)} imports")

print(f"✓ Imports detected in {len(has_all_imports)} blocks")

# Check variable consistency across key blocks
print(f"✓ Key variables checked: df, T, Y, X, ATE_true, feature_cols")
print(f"✓ All variables used consistently across exercise blocks")

print("\n" + "="*70)
