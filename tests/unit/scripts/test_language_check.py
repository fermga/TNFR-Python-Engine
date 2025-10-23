from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "check_language.py"
_spec = importlib.util.spec_from_file_location("tnfr_language_check", MODULE_PATH)
assert _spec and _spec.loader
_language_check = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("tnfr_language_check", _language_check)
_spec.loader.exec_module(_language_check)

DATA_MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "language_policy_data.py"
_data_spec = importlib.util.spec_from_file_location(
    "tnfr_language_policy_data", DATA_MODULE_PATH
)
assert _data_spec and _data_spec.loader
_language_policy_data = importlib.util.module_from_spec(_data_spec)
sys.modules.setdefault("tnfr_language_policy_data", _language_policy_data)
_data_spec.loader.exec_module(_language_policy_data)

_RECENT_KEYWORD_CODES = (
    (101, 106, 101, 109, 112, 108, 111),
    (111, 112, 99, 105, 111, 110, 97, 108, 101, 115),
    (100, 101, 112, 101, 110, 100, 101, 110, 99, 105, 97),
    (99, 111, 109, 112, 97, 116, 105, 98, 105, 108, 105, 100, 97, 100),
    (118, 97, 108, 111, 114, 101, 115),
    (100, 101, 98, 101),
    (114, 101, 99, 111, 109, 112, 117, 116, 97, 114),
    (109, 111, 116, 111, 114),
    (112, 111, 114, 95, 100, 101, 102, 101, 99, 116, 111),
)


def test_default_policy_contains_recent_tokens() -> None:
    expected = {
        token.lower()
        for token in _language_policy_data.decode_keyword_codes(_RECENT_KEYWORD_CODES)
    }
    policy_tokens = {
        word.lower() for word in _language_check.DEFAULT_POLICY.disallowed_keywords
    }
    assert expected.issubset(policy_tokens)


def test_scan_file_flags_recent_keyword(tmp_path) -> None:
    repo_root = tmp_path
    sample = repo_root / "snippet.txt"
    keyword = _language_policy_data.decode_keyword_codes(
        (_RECENT_KEYWORD_CODES[0],)
    )[0]
    sample.write_text(f"Line with {keyword} token", encoding="utf-8")

    violations = _language_check._scan_file(  # type: ignore[attr-defined]
        sample,
        _language_check.DEFAULT_POLICY,
        repo_root,
    )

    assert any(keyword in violation for violation in violations)
