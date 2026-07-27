from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # headless-safe backend, deterministic for tests

from live_plot import LiveTrainingPlot


def test_enabled_on_working_backend():
    plot = LiveTrainingPlot(title="test")
    assert plot.enabled is True


def test_update_appends_data_default_metrics():
    plot = LiveTrainingPlot(title="test")
    plot.update(1, 0.5, 0.8)
    plot.update(2, 0.3, 0.9)

    assert plot.epochs == [1, 2]
    assert plot.history["loss"] == [0.5, 0.3]
    assert plot.history["accuracy"] == [0.8, 0.9]


def test_update_appends_data_three_metrics():
    plot = LiveTrainingPlot(title="test", metrics=("loss", "accuracy", "reward"))
    plot.update(1, 0.5, 0.8, 120.0)
    plot.update(2, 0.3, 0.9, 250.0)

    assert plot.epochs == [1, 2]
    assert plot.history["loss"] == [0.5, 0.3]
    assert plot.history["accuracy"] == [0.8, 0.9]
    assert plot.history["reward"] == [120.0, 250.0]


def test_disabled_when_backend_unavailable():
    with patch("live_plot.plt.subplots", side_effect=RuntimeError("no display")):
        plot = LiveTrainingPlot(title="test")
    assert plot.enabled is False


def test_update_is_noop_when_disabled():
    with patch("live_plot.plt.subplots", side_effect=RuntimeError("no display")):
        plot = LiveTrainingPlot(title="test")
    plot.update(1, 0.5, 0.8)  # must not raise
