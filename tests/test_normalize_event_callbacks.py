from tnfr.callback_utils import _normalize_event_callbacks, CallbackEvent


def test_normalize_event_callbacks_filters_and_normalizes():
    def cb(G, ctx):
        pass

    cbs = {
        CallbackEvent.BEFORE_STEP.value: [("cb", cb)],
        "nope": [("cb", cb)],
    }

    # Non-existent event leaves registry untouched
    _normalize_event_callbacks(cbs, "missing")
    assert set(cbs) == {CallbackEvent.BEFORE_STEP.value, "nope"}

    # Invalid event is removed
    _normalize_event_callbacks(cbs, "nope")
    assert set(cbs) == {CallbackEvent.BEFORE_STEP.value}

    # Valid event is normalized into a mapping
    _normalize_event_callbacks(cbs, CallbackEvent.BEFORE_STEP.value)
    assert set(cbs[CallbackEvent.BEFORE_STEP.value]) == {"cb"}


def test_normalize_event_callbacks_drops_invalid_entries():
    def cb1(G, ctx):
        pass

    cbs = {
        CallbackEvent.BEFORE_STEP.value: [("cb1", cb1), ("bad", 1), object()],
    }

    _normalize_event_callbacks(cbs, CallbackEvent.BEFORE_STEP.value)

    assert set(cbs[CallbackEvent.BEFORE_STEP.value]) == {"cb1"}
