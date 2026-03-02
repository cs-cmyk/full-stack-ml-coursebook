"""Verify all diagrams for Chapter 43 are present and valid"""
import os
from pathlib import Path

diagram_dir = Path(__file__).parent

print("=" * 80)
print("Chapter 43 Diagram Verification")
print("=" * 80)

# Expected diagrams
expected_files = {
    'lora_architecture.png': {
        'type': 'matplotlib',
        'min_size': 50000,  # 50KB minimum
        'description': 'LoRA architecture comparison'
    },
    'rlhf_vs_dpo.png': {
        'type': 'matplotlib',
        'min_size': 80000,  # 80KB minimum
        'description': 'RLHF vs DPO comparison'
    },
    'alignment_pipeline.mmd': {
        'type': 'mermaid',
        'min_size': 100,  # Small text file
        'description': 'Alignment pipeline flowchart'
    }
}

print("\nChecking diagram files...")
all_valid = True

for filename, info in expected_files.items():
    filepath = diagram_dir / filename

    if filepath.exists():
        size = filepath.stat().st_size
        if size >= info['min_size']:
            print(f"✓ {filename:30s} - {size:>8,} bytes - {info['description']}")
        else:
            print(f"✗ {filename:30s} - Too small ({size} bytes < {info['min_size']})")
            all_valid = False
    else:
        print(f"✗ {filename:30s} - MISSING")
        all_valid = False

# Check generator scripts
print("\nChecking generator scripts...")
generators = [
    'generate_lora_architecture.py',
    'generate_rlhf_vs_dpo.py'
]

for gen in generators:
    filepath = diagram_dir / gen
    if filepath.exists():
        print(f"✓ {gen}")
    else:
        print(f"✗ {gen} - MISSING")
        all_valid = False

print("\n" + "=" * 80)
if all_valid:
    print("✓ All diagrams verified successfully!")
else:
    print("✗ Some diagrams are missing or invalid")
print("=" * 80)
