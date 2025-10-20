"""Unit tests for ``_normalize_callback_entry`` handling of sequence inputs."""



from tnfr.callback_utils import _normalize_callback_entry, CallbackSpec


def dummy_cb(G, ctx):
    pass


def test_list_entry():
    entry = ["cb", dummy_cb]
    assert _normalize_callback_entry(entry) == CallbackSpec("cb", dummy_cb)


def test_generator_entry():
    entry = (x for x in ("cb", dummy_cb))
    assert _normalize_callback_entry(entry) == CallbackSpec("cb", dummy_cb)


def test_iterable_conversion_type_error_returns_none():
    class BadIter:
        def __iter__(self):
            return self

        def __next__(self):
            raise TypeError("bad iter")

    entry = BadIter()
    assert _normalize_callback_entry(entry) is None

