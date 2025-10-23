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


def test_default_policy_contains_recent_tokens() -> None:
    expected = {
        ("e" "jemplo"),
        ("o" "pcionales"),
        ("d" "ependencia"),
        ("c" "ompatibilidad"),
        ("v" "alores"),
        ("d" "ebe"),
        ("r" "ecomputar"),
        ("m" "otor"),
        ("p" "or_defecto"),
    }
    policy_tokens = {
        word.lower() for word in _language_check.DEFAULT_POLICY.disallowed_keywords
    }
    assert expected.issubset(policy_tokens)


def test_scan_file_flags_recent_keyword(tmp_path) -> None:
    repo_root = tmp_path
    sample = repo_root / "snippet.txt"
    keyword = "e" "jemplo"
    sample.write_text(f"Line with {keyword} token", encoding="utf-8")

    violations = _language_check._scan_file(  # type: ignore[attr-defined]
        sample,
        _language_check.DEFAULT_POLICY,
        repo_root,
    )

    assert any(keyword in violation for violation in violations)
