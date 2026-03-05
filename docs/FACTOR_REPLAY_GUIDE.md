# TNFR Factorization Replay Guide

**Status**: Draft (2025-11-30)

This guide explains how to replay Paley factorization experiments by consuming the
artifacts emitted by `SpectralPaleyFactorizer`. It focuses on large partition
runs (≥1k partitions) where manifest summaries and compressed file indexes make
it possible to fetch only the subsets you require.

## Artifact taxonomy

| Artifact | Purpose | Location |
|----------|---------|----------|
| `_manifest.json` | Full partition manifest (entries + telemetry) | results/certificates/partitioned/<run>/ |
| `_manifest_summary.json` | Aggregate counts + file index metadata | Same directory |
| `_partition_files.txt.gz` | Optional gzip archive of partition file paths when inline listing exceeds the threshold | Same directory |
| Partition certificates | Per-partition payloads with candidate factors and telemetry | Same directory |

Environment knobs:
- `TNFR_PARTITION_OUTPUT_DIR`: overrides the partition root directory.
- `TNFR_PARTITION_FILELIST_THRESHOLD`: controls when `_partition_files.txt.gz` is emitted
  (default 1000). When the partition count exceeds this threshold, `_manifest.json`
  contains an empty `partition_files` list and references the archive in
  `partition_file_archive`.

## Replay workflow

1. **Locate the run directory** using either CLI output (`partition_artifacts`,
   `partition_manifest`, `partition_manifest_summary`, `partition_file_archive`) or
   programmatic access via `SpectralAnalysisResult` fields.
2. **Read `_manifest_summary.json`**:
   - Inspect `partition_count`, `candidate_stats`, `node_stats`, and
     `boundary_stats` to understand the workload distribution.
   - Use the `file_index` block to determine whether the partition list is inlined
     or archived. For archived runs, note the `archive` and `threshold` fields.
3. **Obtain the partition file list**:
   - If `file_index.inline` is `True`, consume the `partition_files` array inside
     `_manifest.json` directly.
   - Otherwise, download `_partition_files.txt.gz` and stream the newline-delimited
     relative paths. Filter the list (e.g., select partitions with matching IDs) to
     avoid transferring unnecessary files.
4. **Fetch per-partition certificates** using the relative paths and combine them
   with metadata from `entries` (candidate counts, node/boundary sizes, telemetry
   maps) to prioritize downstream processing.
5. **Verify grammar invariants** as you replay partitions—each partition payload
   contains the operator sequence context required to check U1-U6 compliance.

## Sample tooling snippet

```python
from pathlib import Path
import gzip
import json

run_dir = Path("results/certificates/partitioned/certificate_299_1764526545")
summary = json.loads((run_dir / "_manifest_summary.json").read_text())
manifest = json.loads((run_dir / "_manifest.json").read_text())

if summary["file_index"]["inline"]:
    partition_paths = manifest["partition_files"]
else:
    archive_path = run_dir / "_partition_files.txt.gz"
    with gzip.open(archive_path, "rt", encoding="utf-8") as archive:
        partition_paths = [line.strip() for line in archive if line.strip()]

selected = [path for path in partition_paths if path.endswith("_p0.json")]
for relative_path in selected:
    payload = json.loads((run_dir / Path(relative_path).name).read_text())
    # Process payload here
```

## Operational notes

- Keep `_manifest_summary.json` in source control or experiment tracking systems to
  avoid rescanning directories when cataloging runs.
- When mirroring data to another host, copy the summary and archive first; you can
  then selectively fetch the referenced partition files as needed.
- The replay flow maintains TNFR reproducibility guarantees provided seeds and
  operator sequences remain unchanged.
