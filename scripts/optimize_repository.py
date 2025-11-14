#!/usr/bin/env python3
"""
Repository Optimization Script

Performs comprehensive optimization of the TNFR-Python-Engine repository:
- Organizes debug/temporary files
- Fixes common code quality issues  
- Optimizes imports and file structure
- Provides cleanup utilities

Usage:
    python scripts/optimize_repository.py [--dry-run] [--target=all|files|imports|structure]
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Set, Dict, Optional

# Repository root
REPO_ROOT = Path(__file__).parent.parent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RepositoryOptimizer:
    """Main repository optimization class."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.changes_made = []
        
    def log_change(self, description: str) -> None:
        """Log a change that was made or would be made."""
        prefix = "[DRY RUN] " if self.dry_run else ""
        logger.info(f"{prefix}{description}")
        self.changes_made.append(description)
    
    def organize_debug_files(self) -> None:
        """Move debug and temporary files to organized locations."""
        logger.info("🗂️  Organizing debug and temporary files...")
        
        debug_scratch = REPO_ROOT / "debug_scratch"
        
        # Patterns for files to move
        debug_patterns = [
            "debug_*.py",
            "test_*.py",  # Root level test files only
            "analyze_*.py",
            "examine_*.py", 
            "detailed_*.py",
            "generate_*.py",
            "visualize_*.py",
            "prototype_*.py",
            "simple_*.py",
            "quick_*.py",
            "resultados_*.txt",
            "resultados_*.md",
            "*_debug.py"
        ]
        
        # Create debug_scratch directory
        if not self.dry_run:
            debug_scratch.mkdir(exist_ok=True)
        self.log_change(f"Created directory: {debug_scratch}")
        
        moved_count = 0
        for pattern in debug_patterns:
            for file_path in REPO_ROOT.glob(pattern):
                # Skip if it's already in debug_scratch or a subdirectory
                if file_path.is_file() and file_path.parent == REPO_ROOT:
                    if not self.dry_run:
                        shutil.move(str(file_path), str(debug_scratch / file_path.name))
                    self.log_change(f"Moved {file_path.name} to debug_scratch/")
                    moved_count += 1
        
        logger.info(f"✅ Organized {moved_count} debug/temporary files")
    
    def fix_print_statements(self) -> None:
        """Replace print statements with proper logging where appropriate."""
        logger.info("🔧 Fixing print statements in source code...")
        
        fixed_count = 0
        src_files = list((REPO_ROOT / "src").rglob("*.py"))
        
        for py_file in src_files:
            if self._fix_prints_in_file(py_file):
                fixed_count += 1
        
        logger.info(f"✅ Fixed print statements in {fixed_count} files")
    
    def _fix_prints_in_file(self, file_path: Path) -> bool:
        """Fix print statements in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Replace common print patterns with logging
            replacements = [
                ('print("Warning:', 'logger.warning("'),
                ('print("Error:', 'logger.error("'),
                ('print("Info:', 'logger.info("'),
                ('print(f"Warning:', 'logger.warning(f"'),
                ('print(f"Error:', 'logger.error(f"'),
                ('print(f"Info:', 'logger.info(f"'),
            ]
            
            for old, new in replacements:
                if old in content:
                    content = content.replace(old, new)
                    
            # Add logger import if we made changes and it's not present
            if content != original_content and 'logger = logging.getLogger(__name__)' not in content:
                if 'import logging' not in content:
                    # Add logging import after other imports
                    lines = content.split('\n')
                    import_index = 0
                    for i, line in enumerate(lines):
                        if line.startswith('import ') or line.startswith('from '):
                            import_index = i + 1
                    lines.insert(import_index, 'import logging')
                    content = '\n'.join(lines)
                
                # Add logger definition
                if 'logger = logging.getLogger(__name__)' not in content:
                    lines = content.split('\n')
                    # Find a good place to add logger (after imports)
                    insert_index = 0
                    for i, line in enumerate(lines):
                        if not line.strip() or line.startswith('#') or line.startswith('import') or line.startswith('from'):
                            insert_index = i + 1
                        else:
                            break
                    lines.insert(insert_index, '')
                    lines.insert(insert_index + 1, 'logger = logging.getLogger(__name__)')
                    content = '\n'.join(lines)
            
            if content != original_content:
                if not self.dry_run:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                self.log_change(f"Fixed print statements in {file_path.relative_to(REPO_ROOT)}")
                return True
                
        except Exception as e:
            logger.warning(f"Could not process {file_path}: {e}")
            
        return False
    
    def update_phony_targets(self) -> None:
        """Update .PHONY targets in Makefile to include new targets."""
        logger.info("📝 Updating Makefile .PHONY targets...")
        
        makefile = REPO_ROOT / "Makefile"
        if not makefile.exists():
            logger.warning("Makefile not found")
            return
            
        try:
            with open(makefile, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add clean-scratch to .PHONY if not present
            if 'clean-scratch' not in content:
                if '.PHONY:' in content:
                    content = content.replace('.PHONY:', '.PHONY: clean-scratch')
                    self.log_change("Added clean-scratch to .PHONY targets")
                
                # Add the clean-scratch target
                clean_target = '''
clean-scratch:
\t@echo "Cleaning debug/scratch files..."
\t@rm -rf debug_scratch/
\t@echo "Removed debug_scratch directory"
'''
                
                # Insert after the clean target
                if 'clean:' in content and 'clean-scratch:' not in content:
                    # Find position after clean target
                    lines = content.split('\n')
                    insert_index = -1
                    in_clean_target = False
                    
                    for i, line in enumerate(lines):
                        if line.strip() == 'clean:':
                            in_clean_target = True
                        elif in_clean_target and line and not line.startswith('\t'):
                            insert_index = i
                            break
                    
                    if insert_index > 0:
                        lines.insert(insert_index, clean_target.strip())
                        content = '\n'.join(lines)
                        self.log_change("Added clean-scratch target to Makefile")
                
                if not self.dry_run:
                    with open(makefile, 'w', encoding='utf-8') as f:
                        f.write(content)
                        
        except Exception as e:
            logger.warning(f"Could not update Makefile: {e}")
    
    def create_gitignore_entries(self) -> None:
        """Add appropriate entries to .gitignore."""
        logger.info("📝 Updating .gitignore...")
        
        gitignore = REPO_ROOT / ".gitignore"
        entries_to_add = [
            "debug_scratch/",
            "*.tmp",
            "*.temp",
            ".DS_Store",
            "Thumbs.db"
        ]
        
        try:
            if gitignore.exists():
                with open(gitignore, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = ""
            
            added_entries = []
            for entry in entries_to_add:
                if entry not in content:
                    content += f"\n{entry}"
                    added_entries.append(entry)
            
            if added_entries and not self.dry_run:
                with open(gitignore, 'w', encoding='utf-8') as f:
                    f.write(content)
                
            if added_entries:
                self.log_change(f"Added {len(added_entries)} entries to .gitignore")
                
        except Exception as e:
            logger.warning(f"Could not update .gitignore: {e}")
    
    def generate_summary(self) -> None:
        """Generate optimization summary."""
        logger.info("📊 Optimization Summary")
        logger.info("=" * 50)
        
        if not self.changes_made:
            logger.info("No changes were needed - repository is already optimized!")
        else:
            logger.info(f"Total changes: {len(self.changes_made)}")
            for change in self.changes_made:
                logger.info(f"  ✅ {change}")
        
        if self.dry_run:
            logger.info("\nThis was a dry run - no actual changes were made.")
            logger.info("Run without --dry-run to apply changes.")
    
    def optimize_all(self) -> None:
        """Run all optimization steps."""
        logger.info("🚀 Starting comprehensive repository optimization...")
        logger.info(f"Repository root: {REPO_ROOT}")
        
        self.organize_debug_files()
        self.fix_print_statements()
        self.update_phony_targets()
        self.create_gitignore_entries()
        
        self.generate_summary()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--target",
        choices=["all", "files", "imports", "structure"],
        default="all",
        help="Optimization target (default: all)"
    )
    
    args = parser.parse_args()
    
    optimizer = RepositoryOptimizer(dry_run=args.dry_run)
    
    if args.target == "all":
        optimizer.optimize_all()
    elif args.target == "files":
        optimizer.organize_debug_files()
    elif args.target == "imports":
        optimizer.fix_print_statements()
    elif args.target == "structure":
        optimizer.update_phony_targets()
        optimizer.create_gitignore_entries()
    
    logger.info("🎉 Repository optimization complete!")


if __name__ == "__main__":
    main()