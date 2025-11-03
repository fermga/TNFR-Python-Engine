#!/usr/bin/env python3
"""Verify internal references in documentation, notebooks, and scripts.

This script checks that all internal file references in markdown files point to
existing files, ensuring documentation remains consistent and navigable.

Usage:
    python scripts/verify_internal_references.py         # Check and report
    python scripts/verify_internal_references.py --verbose # Report all refs
    python scripts/verify_internal_references.py --ci    # Exit 1 if broken refs found
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


def find_markdown_links(content: str) -> List[Tuple[str, str]]:
    """Extract markdown links from content.
    
    Returns:
        List of (link_text, link_path) tuples.
    """
    # Match [text](path) but not [text](http://...) or [text](#anchor)
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    return re.findall(pattern, content)


def is_external_or_anchor(link_path: str) -> bool:
    """Check if link is external URL or anchor."""
    return link_path.startswith(('http://', 'https://', 'mailto:', '#'))


def verify_references(
    base_dir: Path,
    search_dirs: List[str],
    verbose: bool = False
) -> Tuple[List[dict], List[dict]]:
    """Verify all internal references in markdown files.
    
    Args:
        base_dir: Repository root directory
        search_dirs: List of directories to search for markdown files
        verbose: Print all references if True
        
    Returns:
        Tuple of (all_refs, broken_refs) where each is a list of dicts
    """
    all_refs = []
    broken_refs = []
    
    for search_dir in search_dirs:
        dir_path = base_dir / search_dir
        if not dir_path.exists():
            if verbose:
                print(f"Warning: Directory {search_dir} does not exist")
            continue
        
        for md_file in dir_path.rglob("*.md"):
            try:
                content = md_file.read_text(encoding='utf-8')
            except Exception as e:
                print(f"Error reading {md_file}: {e}")
                continue
            
            links = find_markdown_links(content)
            
            for link_text, link_path in links:
                # Skip external URLs and anchors
                if is_external_or_anchor(link_path):
                    continue
                
                # Handle anchor in local files (e.g., file.md#section)
                clean_path = link_path.split('#')[0] if '#' in link_path else link_path
                if not clean_path:  # Just an anchor
                    continue
                
                # Resolve relative path
                ref_path = (md_file.parent / clean_path).resolve()
                
                ref_info = {
                    'source_file': str(md_file.relative_to(base_dir)),
                    'link_text': link_text,
                    'link_path': link_path,
                    'resolved_path': str(ref_path),
                    'exists': ref_path.exists()
                }
                
                all_refs.append(ref_info)
                
                if not ref_path.exists():
                    broken_refs.append(ref_info)
                elif verbose:
                    print(f"✓ {ref_info['source_file']}: {link_path}")
    
    return all_refs, broken_refs


def main():
    parser = argparse.ArgumentParser(
        description='Verify internal references in TNFR documentation'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print all references, not just broken ones'
    )
    parser.add_argument(
        '--ci',
        action='store_true',
        help='CI mode: exit with code 1 if broken references found'
    )
    parser.add_argument(
        '--dirs',
        nargs='+',
        default=['docs/source', 'examples', 'scripts', '.'],
        help='Directories to search (default: docs/source examples scripts .)'
    )
    
    args = parser.parse_args()
    
    # Get repository root (script is in scripts/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    print(f"Verifying internal references in {repo_root}")
    print(f"Searching directories: {', '.join(args.dirs)}")
    print()
    
    all_refs, broken_refs = verify_references(
        repo_root,
        args.dirs,
        verbose=args.verbose
    )
    
    # Report results
    print(f"Total internal references found: {len(all_refs)}")
    print(f"Broken references: {len(broken_refs)}")
    print()
    
    if broken_refs:
        print("=" * 70)
        print("BROKEN REFERENCES")
        print("=" * 70)
        for ref in broken_refs:
            print(f"\nSource: {ref['source_file']}")
            print(f"  Link text: {ref['link_text']}")
            print(f"  Link path: {ref['link_path']}")
            print(f"  Resolved to: {ref['resolved_path']}")
            print(f"  Status: ✗ NOT FOUND")
        print()
        
        if args.ci:
            print("CI mode: Exiting with error code 1")
            sys.exit(1)
    else:
        print("✓ All internal references are valid!")
        
    return 0


if __name__ == '__main__':
    sys.exit(main())
