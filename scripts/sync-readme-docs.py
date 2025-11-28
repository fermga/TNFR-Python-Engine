#!/usr/bin/env python3
"""
Sync README files to docs directory for MkDocs build.

This script copies the main repository README and module READMEs to the docs
directory so that MkDocs can access them during the build process.
"""

import shutil
from pathlib import Path


def main():
    """Copy README files to docs directory."""
    repo_root = Path(__file__).parent.parent
    docs_dir = repo_root / "docs"
    
    # Ensure docs directory exists
    docs_dir.mkdir(exist_ok=True)
    
    # Copy main README to docs as index
    main_readme = repo_root / "README.md"
    if main_readme.exists():
        shutil.copy2(main_readme, docs_dir / "index.md")
        print(f"✅ Copied {main_readme} → {docs_dir / 'index.md'}")
    
    # Copy important root-level documentation
    root_docs = [
        "DOCUMENTATION_INDEX.md",
        "CANONICAL_SOURCES.md",
        "UNIFIED_GRAMMAR_RULES.md",
        "AGENTS.md",
        "ARCHITECTURE.md",
        "GLOSSARY.md"
    ]
    
    for doc in root_docs:
        source = repo_root / doc
        dest = docs_dir / doc
        if source.exists():
            shutil.copy2(source, dest)
            print(f"✅ Copied {source} → {dest}")
        else:
            print(f"⚠️  Missing: {source}")
    
    # Copy module READMEs
    src_dir = repo_root / "src" / "tnfr"
    modules = [
        "mathematics",
        "physics",
        "operators",
        "dynamics",
        "metrics",
        "sdk",
        "telemetry",
        "tutorials",
        "extensions"
    ]
    
    modules_dir = docs_dir / "modules"
    modules_dir.mkdir(exist_ok=True)
    
    for module in modules:
        source = src_dir / module / "README.md"
        dest = modules_dir / f"{module}.md"
        if source.exists():
            shutil.copy2(source, dest)
            print(f"✅ Copied {source} → {dest}")
        else:
            print(f"⚠️  Missing: {source}")
    
    print("\n🎉 Documentation sync complete!")
    print(f"📁 Files copied to: {docs_dir}")
    print(f"🔗 Module docs in: {modules_dir}")


if __name__ == "__main__":
    main()
