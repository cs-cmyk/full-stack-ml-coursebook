"""
Verification script to ensure all diagram references in content.md are valid
"""
import os
import re

def verify_diagrams():
    content_path = '/home/chirag/ds-book/book/course-02-programming/ch08-sql-databases/content.md'
    diagrams_dir = '/home/chirag/ds-book/book/course-02-programming/ch08-sql-databases/diagrams'

    # Read content.md
    with open(content_path, 'r') as f:
        content = f.read()

    # Find all diagram references
    pattern = r'!\[([^\]]+)\]\(diagrams/([^\)]+)\)'
    references = re.findall(pattern, content)

    print("=" * 70)
    print("DIAGRAM VERIFICATION REPORT")
    print("=" * 70)
    print()

    print(f"Found {len(references)} diagram references in content.md:")
    print()

    all_valid = True
    for i, (description, filename) in enumerate(references, 1):
        filepath = os.path.join(diagrams_dir, filename)
        exists = os.path.exists(filepath)
        status = "✓" if exists else "✗"

        print(f"{i}. {status} {filename}")
        print(f"   Description: {description}")

        if exists:
            size = os.path.getsize(filepath)
            print(f"   File size: {size:,} bytes ({size/1024:.1f} KB)")
        else:
            print(f"   ⚠️  FILE NOT FOUND: {filepath}")
            all_valid = False

        print()

    # List all PNG files in diagrams directory
    png_files = [f for f in os.listdir(diagrams_dir) if f.endswith('.png')]
    print(f"Total PNG files in diagrams directory: {len(png_files)}")

    # Check for unreferenced diagrams
    referenced_files = set(filename for _, filename in references)
    unreferenced = set(png_files) - referenced_files

    if unreferenced:
        print()
        print("Unreferenced diagrams (not used in content.md):")
        for filename in unreferenced:
            print(f"  - {filename}")

    print()
    print("=" * 70)
    if all_valid:
        print("✅ VERIFICATION PASSED: All diagram references are valid!")
    else:
        print("❌ VERIFICATION FAILED: Some diagrams are missing!")
    print("=" * 70)

    return all_valid

if __name__ == '__main__':
    verify_diagrams()
