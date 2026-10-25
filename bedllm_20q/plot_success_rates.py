import math

import matplotlib.pyplot as plt
import numpy as np


def percent_and_standard_error(count, total_items):
    """Return ``(rate, se)`` in percentage points for ``count`` of ``total_items``.

    ``se`` is the standard error of a binomial proportion, scaled to
    percentage points like ``rate``. It is 0 at the two extremes, where the
    proportion has no sampling variance.
    """
    if total_items <= 0:
        raise ValueError(f"total_items must be positive, got {total_items!r}")
    p = count / total_items
    rate = 100.0 * p
    if count == 0 or count == total_items:
        return rate, 0.0
    return rate, 100.0 * math.sqrt(p * (1.0 - p) / total_items)


def plot_success_rates(
    data_with_names,
    *,
    total_items,
    plot_title="Success Rate Over Turns",
    x_label="Turn Number",
    y_label="Success Rate (%)",
    include_error_bands=False,
    filename=None,
):
    """Plot several success-rate curves on one paper-ready figure.

    Args:
        data_with_names: list of ``(dict, name)`` pairs, where each dict maps a
            turn number to the cumulative number of items guessed correctly
            by that turn (the ``_cumulative_correct_by_turn`` output).
        total_items: the number of items each count is out of. Counts are
            converted to percentages here, so the axis is a true rate however
            many items the run played.
        plot_title, x_label, y_label: figure text.
        include_error_bands: draw standard-error bands around each line.
        filename: if given, save the figure here (e.g. "success_rates.pdf")
            and do not open a window; otherwise show it interactively.
    """
    # Set a professional plot style
    plt.style.use("seaborn-v0_8-paper")  # Good for academic papers

    fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes object together

    # Define a color cycle for distinct lines (using a perceptually uniform colormap)
    colors = plt.get_cmap("Dark2", len(data_with_names))
    # Define a cycle of markers and linestyles for differentiation
    markers = [
        "o",
        "s",
        "^",
        "D",
        "v",
        "P",
        "X",
        "*",
    ]  # Circle, square, triangle_up, diamond, etc.
    linestyles = ["-", "--", ":", "-."]  # Solid, dashed, dotted, dash-dotted
    # Determine the overall min and max turn numbers across all datasets for consistent x-axis
    all_turns = []
    for data_dict, _ in data_with_names:
        all_turns.extend(data_dict.keys())

    if not all_turns:
        print("No data provided to plot.")
        return

    min_turn = min(all_turns)
    max_turn = max(all_turns)

    # Plot each dictionary
    for i, (data_dict, name) in enumerate(data_with_names):
        turns = sorted(data_dict.keys())
        rates_and_errors = [
            percent_and_standard_error(data_dict[turn], total_items) for turn in turns
        ]
        success_rates = [rate for rate, _ in rates_and_errors]

        color = colors(i)

        # Plot the main line
        ax.plot(
            turns,
            success_rates,
            color=color,
            marker=markers[i % len(markers)],  # Cycle through markers
            linestyle=linestyles[i % len(linestyles)],  # Cycle through linestyles
            linewidth=2,  # Make lines a bit thicker
            markersize=7,  # Make markers a bit larger
            label=name,
        )

        # Add error bands if requested
        if include_error_bands:
            se_values = [se for _, se in rates_and_errors]
            upper_bound = [
                min(rate + se, 100) for rate, se in zip(success_rates, se_values)
            ]
            lower_bound = [
                max(rate - se, 0) for rate, se in zip(success_rates, se_values)
            ]

            # Plot the error band
            ax.fill_between(turns, lower_bound, upper_bound, color=color, alpha=0.2)

    # Set titles and labels with appropriate font sizes
    ax.set_title(plot_title, fontsize=16, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    # Ensure integer x-axis ticks
    ax.set_xlim(min_turn, max_turn)
    ax.set_xticks(np.arange(min_turn, max_turn + 1, 1))
    ax.tick_params(axis="x", rotation=45, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    # Customize grid
    ax.grid(True, linestyle="--", alpha=0.7)  # Finer, less obtrusive grid

    # Customize legend
    ax.legend(
        loc="upper left", fontsize=12, frameon=True, shadow=True, borderpad=1
    )  # Position, size, border, shadow

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a filename is provided; a batch run must not block
    # on a window, so the interactive show is for the no-filename case only.
    if filename:
        plt.savefig(
            filename, dpi=300, bbox_inches="tight"
        )  # High DPI, tight bounding box
        print(f"Plot saved to {filename}")
        plt.close(fig)
    else:
        plt.show()
