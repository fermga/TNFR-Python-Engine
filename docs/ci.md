# CI Infrastructure Notes

## Virtual Environment Permission Issues

### Problem

When GitHub Actions workflows use artifact upload/download to share virtual environments between jobs, 
the executables in `.venv/bin` may lose their execute permissions. This occurs because:

1. The `actions/upload-artifact@v5` action packages files as-is
2. The `actions/download-artifact@v6` action restores files but may not preserve execute bits
3. This leads to `PermissionError: [Errno 13] Permission denied` when attempting to run tools like `semgrep` or `bandit`

### Solution

The SAST workflow (`.github/workflows/sast-lint.yml`) includes a dedicated step to fix permissions 
before running any tools from the virtual environment:

```yaml
- name: Ensure venv executables are runnable
  run: |
    if [ -d ".venv/bin" ]; then
      chmod -R u+rx .venv/bin || true
      if [ ! -x .venv/bin/semgrep ]; then
        pip install --force-reinstall semgrep
        chmod u+rx .venv/bin/semgrep || true
      fi
    fi
```

This approach:
- Sets read+execute permissions on all binaries in `.venv/bin`
- Gracefully handles missing directories
- Reinstalls semgrep if it's still not executable after chmod
- Uses `|| true` to prevent workflow failures from permission errors

### Helper Script

The repository includes `scripts/fix_venv_perms.sh`, a POSIX-compliant shell script that:
- Checks for the existence of `.venv/bin`
- Sets `u+rx` permissions on all files
- Verifies key executables (python, pip) are executable
- Logs all actions for debugging
- Is idempotent and safe to run multiple times

Usage:
```bash
# Fix default .venv location
./scripts/fix_venv_perms.sh

# Fix custom venv location
./scripts/fix_venv_perms.sh /path/to/venv
```

### Testing

The test suite includes `tests/ci/test_sast_permissions.py` which validates that the permission 
fix script works correctly by:
1. Creating a temporary virtual environment structure
2. Creating a mock executable without execute permissions
3. Running the fix script
4. Asserting the file becomes executable

### References

- Workflow implementation: `.github/workflows/sast-lint.yml`
- GitHub Actions artifact behavior: [actions/upload-artifact#180](https://github.com/actions/upload-artifact/issues/180)

### Best Practices

When working with virtual environments in CI:

1. **Always verify executability** before running venv binaries
2. **Use explicit chmod** after downloading artifacts
3. **Log permission states** for debugging failed runs
4. **Consider alternatives** like caching instead of artifacts for venvs
5. **Test permission fixes** in isolation before deploying to production workflows

### Troubleshooting

If you encounter permission errors:

1. Check the workflow logs for "Permission denied" errors
2. Verify the artifact download step completed successfully
3. Add a debug step to list permissions: `ls -la .venv/bin`
4. Run the fix script manually: `./scripts/fix_venv_perms.sh`
5. Consider reinstalling the problematic tool if chmod alone doesn't fix it
