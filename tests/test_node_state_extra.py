from tnfr.types import NodeState


def test_node_state_extra_access():
    state = NodeState()
    assert state.extra == {}
    state.extra["foo"] = "bar"
    assert state.extra["foo"] == "bar"
