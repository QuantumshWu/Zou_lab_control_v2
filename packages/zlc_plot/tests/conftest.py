from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gui: requires an offscreen Qt backend")


@pytest.fixture(autouse=True)
def close_matplotlib_figures() -> None:
    yield
    plt.close("all")
