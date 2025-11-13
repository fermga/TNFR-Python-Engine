from importlib import import_module
import os
import json


def test_molecule_atlas_outputs(tmp_path):
    mod = import_module('examples.molecule_atlas')
    out_html = mod.main()
    # HTML path returned
    assert os.path.exists(out_html) and out_html.endswith('molecule_atlas.html')
    base = os.path.dirname(out_html)
    # JSONL and CSV should also exist
    jsonl_path = os.path.join(base, 'molecule_atlas.jsonl')
    csv_path = os.path.join(base, 'molecule_atlas.csv')
    assert os.path.exists(jsonl_path)
    assert os.path.exists(csv_path)

    # Basic content checks: JSONL includes 'signature'; CSV header has 'signature'
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        first = f.readline().strip()
        if first:
            obj = json.loads(first)
            assert 'signature' in obj
    with open(csv_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        assert 'signature' in header.split(',')
