import tnfr


def test_public_exports():
    expected = {'__version__', 'step', 'run', 'preparar_red', 'create_nfr', 'NodeState'}
    assert set(tnfr.__all__) == expected


def test_basic_flow():
    G, n = tnfr.create_nfr('n1')
    tnfr.preparar_red(G)
    tnfr.step(G)
    tnfr.run(G, steps=2)
    assert len(G.graph['history']['C_steps']) == 3
    assert isinstance(tnfr.NodeState(), tnfr.NodeState)


def test_removed_name():
    assert not hasattr(tnfr, 'apply_topological_remesh')
