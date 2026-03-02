"""
Diagram Verification Script
Checks that all diagrams are generated and have reasonable properties
"""

from pathlib import Path
from PIL import Image

# Expected diagrams
expected_diagrams = [
    'serialization_comparison.png',
    'docker_layers.png',
    'kubernetes_architecture.png',
    'deployment_strategies.png'
]

# Minimum expected properties
min_width = 600  # pixels
min_height = 400  # pixels
min_size = 50000  # bytes (50 KB)
max_size = 500000  # bytes (500 KB)

print("="*60)
print("Diagram Verification Report")
print("="*60)

script_dir = Path(__file__).parent
all_valid = True

for diagram_name in expected_diagrams:
    diagram_path = script_dir / diagram_name
    print(f"\n{diagram_name}")
    print("-"*60)

    # Check existence
    if not diagram_path.exists():
        print(f"  ✗ File not found!")
        all_valid = False
        continue
    else:
        print(f"  ✓ File exists")

    # Check file size
    file_size = diagram_path.stat().st_size
    size_kb = file_size / 1024
    print(f"  ✓ File size: {size_kb:.1f} KB", end="")

    if file_size < min_size:
        print(f" (WARNING: smaller than {min_size/1024:.0f} KB)")
        all_valid = False
    elif file_size > max_size:
        print(f" (WARNING: larger than {max_size/1024:.0f} KB)")
    else:
        print()

    # Check image properties
    try:
        with Image.open(diagram_path) as img:
            width, height = img.size
            mode = img.mode
            print(f"  ✓ Dimensions: {width} x {height} pixels")
            print(f"  ✓ Color mode: {mode}")

            if width < min_width or height < min_height:
                print(f"  ✗ WARNING: Image too small (minimum {min_width}x{min_height})")
                all_valid = False

            # Check DPI if available
            dpi = img.info.get('dpi', None)
            if dpi:
                print(f"  ✓ DPI: {dpi}")
            else:
                print(f"  - DPI: Not specified")

    except Exception as e:
        print(f"  ✗ Error reading image: {e}")
        all_valid = False

print("\n" + "="*60)
if all_valid:
    print("✓ All diagrams verified successfully!")
    print("="*60)
    exit(0)
else:
    print("✗ Some diagrams have issues - please review warnings above")
    print("="*60)
    exit(1)
