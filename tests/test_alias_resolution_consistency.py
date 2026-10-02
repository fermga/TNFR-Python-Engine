"""Canonical attribute precedence must not depend on earlier reads."""

import pytest

from tnfr.alias import AliasAccessor, collect_attr, get_attr, set_attr
from tnfr.constants.aliases import ALIAS_VF


@pytest.mark.parametrize("operation", ["get", "set"])
def test_same_size_mapping_edit_restores_first_alias_priority(operation):
    accessor = AliasAccessor(conv=float)
    aliases = ("primary", "fallback")
    data = {"fallback": 2.0, "padding": None}
    assert accessor.get(data, aliases) == 2.0
    del data["padding"]
    data["primary"] = 1.0
    if operation == "get":
        assert accessor.get(data, aliases) == 1.0
    else:
        accessor.set(data, aliases, 3.0)
        assert data == {"primary": 3.0, "fallback": 2.0}


def test_repaired_primary_value_replaces_previously_valid_fallback():
    accessor = AliasAccessor(conv=float)
    data = {"primary": "invalid", "fallback": 2.0}
    assert accessor.get(data, ("primary", "fallback")) == 2.0
    data["primary"] = 1.0
    assert accessor.get(data, ("primary", "fallback")) == 1.0


def test_converter_change_rechecks_higher_priority_alias():
    accessor = AliasAccessor(conv=float)
    data = {"primary": "not a number", "fallback": 2.0}
    assert accessor.get(data, ("primary", "fallback")) == 2.0
    assert accessor.get(data, ("primary", "fallback"), conv=str) == "not a number"


def test_strict_read_does_not_bypass_invalid_primary_after_permissive_read():
    accessor = AliasAccessor(conv=float)
    data = {"primary": "invalid", "fallback": 2.0}
    assert accessor.get(data, ("primary", "fallback")) == 2.0
    with pytest.raises(ValueError):
        accessor.get(data, ("primary", "fallback"), strict=True)


def test_write_precedence_is_independent_of_prior_conversion_fallback():
    accessor = AliasAccessor(conv=float)
    data = {"primary": "invalid", "fallback": 2.0}
    assert accessor.get(data, ("primary", "fallback")) == 2.0
    accessor.set(data, ("primary", "fallback"), 3.0)
    assert data == {"primary": 3.0, "fallback": 2.0}


def test_actual_frequency_alias_read_and_write_follow_canonical_order():
    primary, secondary = ALIAS_VF[:2]
    data = {secondary: 2.0, "padding": 0.0}
    assert get_attr(data, ALIAS_VF) == 2.0
    del data["padding"]
    data[primary] = 1.0
    assert get_attr(data, ALIAS_VF) == 1.0
    set_attr(data, ALIAS_VF, 3.0)
    assert data[primary] == 3.0
    assert data[secondary] == 2.0


def test_existing_alias_write_and_new_mapping_default_remain_compatible():
    accessor = AliasAccessor(conv=float, default=4.0)
    aliases = ("primary", "fallback")
    assert accessor.get({}, aliases) == 4.0
    existing = {"fallback": 2.0}
    accessor.set(existing, aliases, 3.0)
    assert existing == {"fallback": 3.0}
    empty = {}
    accessor.set(empty, aliases, 3.0)
    assert empty == {"primary": 3.0}


def test_temporary_mapping_reads_do_not_grow_an_identity_cache():
    accessor = AliasAccessor(conv=float)
    mappings = [{"primary": float(index)} for index in range(1000)]
    for data in mappings:
        accessor.get(data, ("primary", "fallback"))
    assert not getattr(accessor, "_key_cache", {})


def test_collect_attr_materializes_alias_iterator_once_for_all_nodes():
    import networkx as nx

    graph = nx.Graph()
    graph.add_node("first", value=1.0)
    graph.add_node("second", value=2.0)
    values = collect_attr(graph, iter(graph), iter(["value"]))
    assert list(values) == [1.0, 2.0]
