import pytest

from tnfr import _is_internal_import_error


def test_circular_import_message_with_base_package_name_counts_as_internal():
    exc = ImportError(
        "cannot import name 'create_nfr' from partially initialized module 'tnfr' "
        "(most likely due to a circular import)"
    )

    assert _is_internal_import_error(exc) is True
