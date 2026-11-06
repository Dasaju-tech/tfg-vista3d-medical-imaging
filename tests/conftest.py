from __future__ import annotations

import pytest


# These filters are intentionally narrow. They silence only known deprecation
# warnings emitted by third-party packages used by this project. New warnings
# from our own code, or unrelated dependency warnings, remain visible.
KNOWN_THIRD_PARTY_WARNING_FILTERS = (
    (
        r"ignore:`torch\.jit\.interface` is deprecated.*:"
        r"DeprecationWarning:torch\.jit\._script"
    ),
    (
        r"ignore:Please import .* from the `scipy\.ndimage` namespace.*:"
        r"DeprecationWarning:surface_distance\.metrics"
    ),
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        for warning_filter in KNOWN_THIRD_PARTY_WARNING_FILTERS:
            item.add_marker(
                pytest.mark.filterwarnings(warning_filter)
            )
