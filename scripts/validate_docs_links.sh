#!/bin/bash
# validate_docs_links.sh - Validate internal documentation links

set -e

echo "=== TNFR Documentation Link Validator ==="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0
CHECKED=0

# Increment functions to avoid subshell issues
increment_checked() {
    CHECKED=$((CHECKED + 1))
}

increment_errors() {
    ERRORS=$((ERRORS + 1))
}

increment_warnings() {
    WARNINGS=$((WARNINGS + 1))
}

# Function to check if a file exists
check_file_exists() {
    local file=$1
    local context=$2
    
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $context: $file exists"
        increment_checked
        return 0
    else
        echo -e "${RED}✗${NC} $context: $file NOT FOUND"
        increment_errors
        return 1
    fi
}

echo "Checking Getting Started files..."
check_file_exists "docs/source/getting-started/README.md" "Getting Started landing"
check_file_exists "docs/source/getting-started/FAQ.md" "FAQ"
check_file_exists "docs/source/getting-started/quickstart.md" "Quickstart"
check_file_exists "docs/source/getting-started/TNFR_CONCEPTS.md" "TNFR Concepts"
check_file_exists "docs/source/getting-started/INTERACTIVE_TUTORIAL.md" "Interactive Tutorial"

echo
echo "Checking User Guide files..."
check_file_exists "docs/source/user-guide/OPERATORS_GUIDE.md" "Operators Guide"
check_file_exists "docs/source/user-guide/METRICS_INTERPRETATION.md" "Metrics Interpretation"
check_file_exists "docs/source/user-guide/TROUBLESHOOTING.md" "Troubleshooting"

echo
echo "Checking Advanced files..."
check_file_exists "docs/source/advanced/PERFORMANCE_OPTIMIZATION.md" "Performance Optimization"
check_file_exists "docs/source/advanced/THEORY_DEEP_DIVE.md" "Theory Deep Dive"

echo
echo "Checking API files..."
check_file_exists "docs/source/api/overview.md" "API Overview"
check_file_exists "docs/source/api/operators.md" "API Operators"
check_file_exists "docs/source/api/telemetry.md" "API Telemetry"

echo
echo "Checking Examples..."
check_file_exists "docs/source/examples/README.md" "Examples Index"
check_file_exists "docs/source/examples/controlled_dissonance.py" "Controlled Dissonance Example"

echo
echo "Checking main navigation files..."
check_file_exists "docs/source/index.rst" "Main Index (Sphinx)"
check_file_exists "mkdocs.yml" "MkDocs Config"

echo
echo "=== Link Validation in Markdown Files ==="
echo

# Check for broken relative links in newly created files
for file in docs/source/getting-started/*.md \
            docs/source/user-guide/*.md \
            docs/source/advanced/*.md; do
    
    if [ ! -f "$file" ]; then
        continue
    fi
    
    echo "Checking links in: $file"
    
    # Extract markdown links [text](url)
    grep -oP '\[.*?\]\(\K[^)]+' "$file" | while read -r link; do
        # Skip external links (http, https, mailto)
        if [[ "$link" =~ ^https?:// ]] || [[ "$link" =~ ^mailto: ]]; then
            continue
        fi
        
        # Skip anchors within same file
        if [[ "$link" =~ ^# ]]; then
            continue
        fi
        
        # Get directory of current file
        dir=$(dirname "$file")
        
        # Resolve relative path
        target="$dir/$link"
        
        # Normalize path (remove ..)
        target=$(realpath -m "$target" 2>/dev/null || echo "$target")
        
        # Check if target exists (handle .html → .md conversion)
        target_md="${target%.html}.md"
        target_rst="${target%.html}.rst"
        
        if [ -f "$target" ] || [ -f "$target_md" ] || [ -f "$target_rst" ]; then
            echo -e "  ${GREEN}✓${NC} $link"
            increment_checked
        else
            # Check if it's a directory index
            if [ -d "$target" ] && [ -f "$target/README.md" ]; then
                echo -e "  ${GREEN}✓${NC} $link (directory)"
                increment_checked
            else
                echo -e "  ${YELLOW}⚠${NC} $link (target may not exist)"
                increment_warnings
            fi
        fi
    done
done

echo
echo "=== Summary ==="
echo -e "Files checked: ${GREEN}$CHECKED${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
echo -e "Errors: ${RED}$ERRORS${NC}"

if [ $ERRORS -gt 0 ]; then
    echo
    echo -e "${RED}Documentation validation FAILED${NC}"
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo
    echo -e "${YELLOW}Documentation validation completed with warnings${NC}"
    exit 0
else
    echo
    echo -e "${GREEN}Documentation validation PASSED${NC}"
    exit 0
fi
