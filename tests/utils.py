import tnfr.json_utils as json_utils


def clear_orjson_cache() -> None:
    """Clear cached :mod:`orjson` module and warning state."""
    json_utils._warn_orjson_params_once.clear()
    cache_clear = getattr(json_utils.cached_import, "cache_clear", None)
    if cache_clear:
        cache_clear()
