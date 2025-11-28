
import pandas as pd
from pathlib import Path


def test_manifest_and_csv_exist():
    guide = Path('docs/INTERACTIONS_GUIDE.md')
    assert guide.exists(), 'Guide must exist after running the notebook export cells.'
    # Find at least one manifest and CSV under assets
    assets = Path('docs/assets/interactions')
    assert assets.exists(), 'Assets dir must exist.'
    found_manifest = False
    found_csv = False
    for p in assets.rglob('manifest.json'):
        found_manifest = True
    for p in assets.rglob('telemetry.csv'):
        df = pd.read_csv(p)
        assert not df.empty
        found_csv = True
    assert found_manifest, 'At least one manifest.json must be exported.'
    assert found_csv, 'At least one telemetry.csv must be exported.'
    