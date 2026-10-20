# Retired advanced arithmetic adapters

This directory is a recovery record, not an executable package, benchmark or
test suite. The active application evaluates static arithmetic statistics
through `tnfr_primality.core`, its bounded cache and the basic CLI.

The [manifest](retirement-manifest.json) identifies removed committed sources
by Git revision and blob. Their Git text has LF line endings; the exact CRLF
working bytes are also retained in the external cleanup backup recorded there.
Unique uncommitted work is preserved here byte for byte with `.txt` suffixes:

- [Advanced example](examples/advanced_examples.py.txt).
- [Mixed example tests](tests/test_examples.py.txt); the maintained test keeps
  the basic example controls.

The archived example depends on retired adapters and is not a current usage
recipe. Its claims and fallback behavior are not current contracts. Recover a
complete old implementation from the recorded revision when historical
inspection requires it; do not import these files into current execution.

The package whitelist and `MANIFEST.in` exclude this directory from wheel and
source distributions. The root source repository retains this recovery record.
