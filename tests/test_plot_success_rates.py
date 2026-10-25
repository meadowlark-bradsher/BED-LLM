"""The success-rate plot converts counts to percentages with an explicit total.

The old code plotted raw counts on a percent axis and computed error bands
as if every count were out of 100, which was only right because each
dataset happens to hold 100 items.
"""

import math
from types import SimpleNamespace as NS

import matplotlib.pyplot as plt
import pytest

import eig_game_play as egp
import plot_success_rates as psr


@pytest.mark.parametrize(
    "count,total,rate,se",
    [
        (50, 100, 50.0, 5.0),
        (1, 1, 100.0, 0.0),
        (0, 4, 0.0, 0.0),
        (1, 4, 25.0, 100.0 * math.sqrt(0.25 * 0.75 / 4)),
        (3, 10, 30.0, 100.0 * math.sqrt(0.3 * 0.7 / 10)),
    ],
)
def test_percent_and_standard_error(count, total, rate, se):
    got_rate, got_se = psr.percent_and_standard_error(count, total)
    assert got_rate == pytest.approx(rate)
    assert got_se == pytest.approx(se)


def test_percent_and_standard_error_rejects_a_non_positive_total():
    with pytest.raises(ValueError, match="total_items must be positive"):
        psr.percent_and_standard_error(0, 0)


@pytest.fixture
def plotted(monkeypatch):
    """Capture every line plotted and whether a window was requested."""
    lines = []
    shown = []
    real_plot = plt.Axes.plot

    def plot(self, x, y, *args, **kwargs):
        lines.append((list(x), list(y), kwargs.get("label")))
        return real_plot(self, x, y, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "plot", plot)
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    return NS(lines=lines, shown=shown)


def test_plot_scales_counts_by_total_items_and_saves_without_showing(tmp_path, plotted):
    out = tmp_path / "rates.pdf"
    # Two items, one guessed at turn 1 and the other at turn 3.
    psr.plot_success_rates(
        [({1: 1, 2: 1, 3: 2}, "EIG")],
        total_items=2,
        include_error_bands=True,
        filename=str(out),
    )
    assert out.exists() and out.stat().st_size > 0
    assert plotted.shown == []
    (turns, rates, label), = plotted.lines
    assert label == "EIG"
    assert turns == [1, 2, 3]
    assert rates == pytest.approx([50.0, 50.0, 100.0])


def test_plot_shows_interactively_only_without_a_filename(plotted):
    psr.plot_success_rates([({1: 0, 2: 1}, "x")], total_items=1)
    assert plotted.shown == [True]


def test_make_plots_passes_the_item_count(monkeypatch, run_config):
    captured = {}
    monkeypatch.setattr(egp, "plot_success_rates", lambda **kw: captured.update(kw))
    cfg = run_config("--model_name=gpt-4.1", "--dataset=animals")
    items = ["Lion", "Tiger", "Bear"]
    results = [
        egp.AcquisitionResult(
            acquisition="EIG", items=items, successes=[True, False, True],
            cumulative_correct_by_turn={1: 1, 2: 2}, theta_payload={}, correct_count=2,
        )
    ]
    egp._make_plots(results, egp.ArtifactPaths(cfg, "ts"), cfg)
    assert captured["total_items"] == 3
    assert captured["data_with_names"] == [({1: 1, 2: 2}, "EIG")]
    assert captured["filename"].endswith("success_rates_gpt-4.1_gpt-4.1_animals.pdf")
