#!/usr/bin/env bash
# run_pip_audit.sh - Run pip-audit on TNFR Python Engine dependencies
#
# This script runs pip-audit to scan for known security vulnerabilities
# in installed Python dependencies. It mimics the behavior of the CI/CD
# pipeline's pip-audit workflow.
#
# Usage:
#   ./scripts/run_pip_audit.sh [--install] [--json] [--help]
#
# Options:
#   --install    Install pip-audit if not already available
#   --json       Output results in JSON format to pip-audit.json
#   --help       Show this help message
#
# Examples:
#   # Run audit with default settings
#   ./scripts/run_pip_audit.sh
#
#   # Install pip-audit and run audit
#   ./scripts/run_pip_audit.sh --install
#
#   # Generate JSON report
#   ./scripts/run_pip_audit.sh --json

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
INSTALL_AUDIT=false
JSON_OUTPUT=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --install)
            INSTALL_AUDIT=true
            shift
            ;;
        --json)
            JSON_OUTPUT=true
            shift
            ;;
        --help|-h)
            # Extract help from comments between line 2 and the first line that doesn't start with #
            awk '/^# /{if(NR>1)print substr($0,3)}; /^[^#]/{exit}' "$0"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=== TNFR Python Engine - Dependency Vulnerability Audit ===${NC}\n"

# Check if pip-audit is installed
if ! command -v pip-audit &> /dev/null; then
    if [ "$INSTALL_AUDIT" = true ]; then
        echo -e "${YELLOW}Installing pip-audit...${NC}"
        # Check if we're in a virtual environment
        if [ -n "$VIRTUAL_ENV" ] || [ -n "$CONDA_DEFAULT_ENV" ]; then
            # In virtual environment, install without --user
            python -m pip install pip-audit
        else
            # Outside virtual environment, use --user to avoid system-wide installation
            python -m pip install --user pip-audit
        fi
        echo -e "${GREEN}pip-audit installed successfully${NC}\n"
    else
        echo -e "${RED}Error: pip-audit is not installed${NC}"
        echo -e "Install it with: ${YELLOW}pip install pip-audit${NC}"
        echo -e "Or run this script with: ${YELLOW}$0 --install${NC}"
        exit 1
    fi
fi

# Get site-packages path
SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")
echo -e "${BLUE}Scanning installed packages in:${NC} $SITE_PACKAGES\n"

# Run pip-audit
OUTPUT_FILE="pip-audit-$(date +%s)-$$.json"

if [ "$JSON_OUTPUT" = true ]; then
    echo -e "${BLUE}Running pip-audit (JSON output)...${NC}\n"
    if pip-audit --progress-spinner off --format json --output "$OUTPUT_FILE" --path "$SITE_PACKAGES"; then
        echo -e "\n${GREEN}✓ No vulnerabilities found${NC}"
        echo -e "${BLUE}JSON report saved to:${NC} $OUTPUT_FILE"
        exit 0
    else
        echo -e "\n${RED}✗ Vulnerabilities detected${NC}"
        echo -e "${BLUE}JSON report saved to:${NC} $OUTPUT_FILE"
        echo -e "\n${YELLOW}Review the report and update vulnerable dependencies in pyproject.toml${NC}"
        echo -e "${YELLOW}See SECURITY.md for the security update process${NC}"
        exit 1
    fi
else
    echo -e "${BLUE}Running pip-audit (human-readable output)...${NC}\n"
    if pip-audit --progress-spinner off --path "$SITE_PACKAGES"; then
        echo -e "\n${GREEN}✓ No vulnerabilities found${NC}"
        exit 0
    else
        echo -e "\n${RED}✗ Vulnerabilities detected${NC}"
        echo -e "\n${YELLOW}To generate a detailed JSON report, run:${NC}"
        echo -e "  ${BLUE}$0 --json${NC}"
        echo -e "\n${YELLOW}See SECURITY.md for the security update process${NC}"
        exit 1
    fi
fi
