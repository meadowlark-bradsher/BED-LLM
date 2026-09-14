import matplotlib.pyplot as plt
import numpy as np


def plot_success_rates(
    data_with_names,
    plot_title="Success Rate Over Turns",
    x_label="Turn Number",
    y_label="Success Rate (%)",
    include_error_bands=False,
    filename=None,
):
    """Plot several success-rate curves on one paper-ready figure.

    Args:
        data_with_names: list of ``(dict, name)`` pairs, where each dict maps a
            turn number to a success percentage.
        plot_title, x_label, y_label: figure text.
        include_error_bands: draw standard-error bands around each line.
        filename: if given, save the figure here (e.g. "success_rates.pdf").
    """
    # Function to compute standard error
    def compute_standard_error(n, total=100):
        """Compute standard error for a binomial proportion"""
        if n == 0 or n == total:
            return 0
        p = n / total
        return np.sqrt(p * (1 - p) / total)

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
        success_rates = [data_dict[turn] for turn in turns]

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
            # Calculate standard errors
            se_values = [compute_standard_error(rate) * 100 for rate in success_rates]
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

    # Save the figure if a filename is provided
    if filename:
        plt.savefig(
            filename, dpi=300, bbox_inches="tight"
        )  # High DPI, tight bounding box
        print(f"Plot saved to {filename}")

    plt.show()
