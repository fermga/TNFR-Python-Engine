#!/bin/sh
# fix_venv_perms.sh - Ensure virtual environment binaries are executable
#
# This script fixes a common issue in CI environments where virtual environment
# binaries may lose execute permissions during artifact upload/download cycles.
# The script is idempotent and safe to run multiple times.
#
# Usage:
#   ./scripts/fix_venv_perms.sh [venv_path]
#
# Arguments:
#   venv_path: Path to the virtual environment (default: .venv)
#
# Exit codes:
#   0 - Success (permissions fixed or already correct)
#   1 - Error (venv directory not found)

set -e

# Default venv path
VENV_PATH="${1:-.venv}"
BIN_DIR="${VENV_PATH}/bin"

echo "fix_venv_perms: Checking virtual environment at ${VENV_PATH}"

# Check if venv bin directory exists
if [ ! -d "${BIN_DIR}" ]; then
    echo "fix_venv_perms: ERROR - Virtual environment bin directory not found: ${BIN_DIR}"
    exit 1
fi

# Count files that need permission fixes
count=0
for file in "${BIN_DIR}"/*; do
    if [ -f "${file}" ]; then
        if [ ! -x "${file}" ]; then
            count=$((count + 1))
        fi
    fi
done

echo "fix_venv_perms: Found ${count} file(s) needing execute permissions"

# Fix permissions on all files in bin directory
# Use u+rx to ensure user can read and execute
if chmod -R u+rx "${BIN_DIR}" 2>/dev/null; then
    echo "fix_venv_perms: Successfully set u+rx permissions on ${BIN_DIR}"
else
    echo "fix_venv_perms: WARNING - chmod failed, but continuing (may already have correct permissions)"
fi

# Verify key executables are now executable
key_executables="python python3 pip pip3"
missing_count=0
for exe in ${key_executables}; do
    exe_path="${BIN_DIR}/${exe}"
    if [ -f "${exe_path}" ]; then
        if [ -x "${exe_path}" ]; then
            echo "fix_venv_perms: ✓ ${exe} is executable"
        else
            echo "fix_venv_perms: ✗ ${exe} is NOT executable (permission fix may have failed)"
            missing_count=$((missing_count + 1))
        fi
    fi
done

if [ ${missing_count} -gt 0 ]; then
    echo "fix_venv_perms: WARNING - ${missing_count} key executable(s) still not executable"
else
    echo "fix_venv_perms: All key executables verified"
fi

echo "fix_venv_perms: Done"
exit 0
