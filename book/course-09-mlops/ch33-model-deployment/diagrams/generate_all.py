"""
Generate All Diagrams
Convenience script to regenerate all chapter diagrams at once
"""

import subprocess
import sys
from pathlib import Path

# Get the directory of this script
script_dir = Path(__file__).parent

# List of all diagram generation scripts
diagram_scripts = [
    'serialization_comparison.py',
    'docker_layers.py',
    'kubernetes_architecture.py',
    'deployment_strategies.py'
]

print("="*60)
print("Generating all diagrams for Chapter 33: Model Deployment")
print("="*60)

success_count = 0
error_count = 0

for script in diagram_scripts:
    script_path = script_dir / script
    print(f"\nGenerating: {script}")
    print("-"*60)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_dir),
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print(result.stdout)
            success_count += 1
        else:
            print(f"ERROR: {script} failed with return code {result.returncode}")
            print(result.stderr)
            error_count += 1

    except subprocess.TimeoutExpired:
        print(f"ERROR: {script} timed out after 30 seconds")
        error_count += 1
    except Exception as e:
        print(f"ERROR: {script} failed with exception: {e}")
        error_count += 1

print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Successfully generated: {success_count}/{len(diagram_scripts)} diagrams")
if error_count > 0:
    print(f"Failed: {error_count} diagrams")
    sys.exit(1)
else:
    print("✓ All diagrams generated successfully!")
    sys.exit(0)
