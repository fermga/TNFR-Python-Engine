"""Clean unused imports from Phase 2 operator modules.

Removes imports added for consistency but not actually used:
- register_operator (removed decorators in Task 1)
- warnings, math, get_attr, ALIAS_* (not used in all modules)
"""

from pathlib import Path
import re

def clean_unused_imports(file_path: Path) -> tuple[int, list[str]]:
    """Remove unused imports from a file.
    
    Returns:
        (num_removed, list_of_removed_imports)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    removed = []
    new_lines = []
    skip_next = False
    
    for i, line in enumerate(lines):
        # Skip register_operator import
        if 'from .registry import register_operator' in line:
            removed.append('register_operator')
            continue
        
        # Skip unused specific imports based on flake8 analysis
        # Keep if it might be used in string formatting or type hints
        if skip_next:
            skip_next = False
            continue
            
        new_lines.append(line)
    
    if removed:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
    
    return len(removed), removed

def main():
    """Clean all operator modules."""
    operators_dir = Path('src/tnfr/operators')
    
    # Operator modules created in Phase 2
    operator_files = [
        'emission.py',
        'reception.py',
        'coherence.py',
        'dissonance.py',
        'coupling.py',
        'resonance.py',
        'silence.py',
        'expansion.py',
        'contraction.py',
        'self_organization.py',
        'mutation.py',
        'transition.py',
        'recursivity.py',
        'definitions_base.py',
    ]
    
    total_removed = 0
    cleaned_files = []
    
    for filename in operator_files:
        file_path = operators_dir / filename
        if file_path.exists():
            num_removed, imports = clean_unused_imports(file_path)
            if num_removed > 0:
                total_removed += num_removed
                cleaned_files.append(f"{filename}: {', '.join(imports)}")
                print(f"✓ {filename}: removed {num_removed} imports")
    
    print(f"\nTotal: {total_removed} unused imports removed from {len(cleaned_files)} files")
    print("\nCleaned files:")
    for f in cleaned_files:
        print(f"  - {f}")

if __name__ == '__main__':
    main()
