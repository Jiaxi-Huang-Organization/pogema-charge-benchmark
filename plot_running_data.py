"""
Plot running data from tensorboard event files.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator


def custom_palette():
    """Custom color palette matching compare_environments.py style."""
    q = list(sns.color_palette("deep"))
    q[1], q[9] = q[9], q[1]
    q[5], q[9] = q[9], q[5]
    q[6], q[9] = q[9], q[6]
    return q


def load_tensorboard_data(event_path: str, tag: str):
    """Load scalar data from a tensorboard event file."""
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    scalars = ea.Scalars(tag)
    steps = [s.step for s in scalars]
    values = [s.value for s in scalars]
    return steps, values


def plot_runs(
    data_dir: str = "charge_running_data",
    tag: str = "reward/reward",
    runs: list = None,
    run_names: dict = None,
    save_path: str = None,
    figsize: tuple = (3.5, 2.6),
    window_size: int = 50,
    title: str = None,
    font_size: int = 8,
    legend_font_size: int = 8,
    line_width: float = 0.4,
):
    """
    Plot running data from multiple tensorboard runs.
    Style matches compare_environments.py.
    """
    data_path = Path(data_dir)

    # Find all runs
    all_runs = {}
    for subdir in sorted(data_path.iterdir()):
        if not subdir.is_dir():
            continue
        events = list(subdir.glob('events.out.tfevents.*'))
        if events:
            all_runs[subdir.name] = events[0]

    if runs is None:
        runs = list(all_runs.keys())

    plt.style.use('seaborn-v0_8-colorblind')
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = font_size
    plt.rcParams['legend.fontsize'] = legend_font_size
    plt.rcParams['figure.facecolor'] = '#FFFFFF'

    fig, ax = plt.subplots()

    colors = custom_palette()[:len(runs)]

    for i, run_name in enumerate(runs):
        if run_name not in all_runs:
            print(f"  Warning: Run '{run_name}' not found, skipping...")
            continue

        event_path = all_runs[run_name]
        steps, values = load_tensorboard_data(str(event_path), tag)

        # Apply moving average smoothing
        if window_size > 1:
            smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            smoothed_steps = steps[window_size//2:len(smoothed) + window_size//2]
        else:
            smoothed = values
            smoothed_steps = steps

        display_name = run_names.get(run_name, run_name) if run_names else run_name

        # Plot smoothed line
        ax.plot(smoothed_steps, smoothed, color=colors[i],
                label=display_name, linewidth=line_width*5)

        # Plot raw data with low alpha
        ax.plot(steps, values, color=colors[i], alpha=0.15, linewidth=line_width)

    ax.set_xlabel('Training Steps')
    ylabel = tag.split('/')[-1].replace('_', ' ').title()
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    legend = ax.legend(loc='best', fontsize=legend_font_size)
    for line in legend.get_lines():
        line.set_linewidth(2.0)  # 图例线宽独立设置
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def get_all_scalar_tags(data_dir: str):
    """Get all available scalar tags from the runs."""
    data_path = Path(data_dir)
    all_tags = set()

    for subdir in sorted(data_path.iterdir()):
        if not subdir.is_dir():
            continue
        events = list(subdir.glob('events.out.tfevents.*'))
        if events:
            ea = event_accumulator.EventAccumulator(str(events[0]))
            ea.Reload()
            all_tags.update(ea.Tags()['scalars'])

    return sorted(all_tags)


def main():
    data_dir = "charge_running_data"
    save_dir = "running_plot"

    # Create save directory
    Path(save_dir).mkdir(exist_ok=True)

    # Define display names for runs
    run_names = {
        'full': 'full',
        'wo_battery_reward': 'w/o battery reward',
        'wo_death_penalty': 'w/o death penalty',
        'wo_enc': 'w/o enc',
        'new_obs': 'w/o position'
    }

    # Get all available tags
    print("Getting available tags...")
    all_tags = get_all_scalar_tags(data_dir)

    # Key tags to plot
    key_tags = [
        'agents_depletion_rate',
        'avg_battery_reward',
        'avg_charging_per_agent',
        'avg_charging_rate',
        'avg_early_death_penalty',
        'avg_goal_battery_relative',
        'avg_intrinsic_reward',
        'avg_position_reward',
        'avg_relative_battery',
        'avg_throughput',
        'avg_throughput_with_active',
        'valid_episode_relative',
    ]

    # Filter to only existing tags
    tags_to_plot = [tag for tag in key_tags if tag in all_tags]

    print(f"Found {len(all_tags)} scalar tags")
    print(f"Will plot {len(tags_to_plot)} key tags:")
    for tag in tags_to_plot:
        print(f"  - {tag}")

    print()
    print("Generating plots...")

    # Plot each tag
    for tag in tags_to_plot:
        save_name = tag.replace('/', '_').replace(' ', '_')
        save_path = Path(save_dir) / f"{save_name}.pdf"

        plot_runs(
            data_dir=data_dir,
            tag=tag,
            run_names=run_names,
            save_path=str(save_path),
            window_size=200,
            font_size=8,
            legend_font_size=8,
            line_width=0.2,
        )

    print()
    print(f"Done! All plots saved to {save_dir}/")


if __name__ == "__main__":
    main()
