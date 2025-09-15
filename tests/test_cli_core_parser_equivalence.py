from __future__ import annotations

from tnfr.cli import _parse_tokens
from tnfr.token_parser import _parse_tokens as core_parse_tokens


def test_cli_and_core_parsers_share_behavior():
    tokens = [{"WAIT": 1}, {"TARGET": "A"}]
    assert list(_parse_tokens(tokens)) == list(core_parse_tokens(tokens))
