"""
Ablation experiment plotting with radar/polygon charts.
"""

import json
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import yaml


def load_json_results(folder_path: Path) -> List[dict]:
    """Load all JSON result files from a folder."""
    results = []
    for file in folder_path.glob('*.json'):
        with open(file, 'r') as f:
            results.extend(json.load(f))
    return results


def extract_metrics(results: List[dict]) -> pd.DataFrame:
    """Convert results list to DataFrame."""
    data = {}
    for idx, config in enumerate(results):
        row = {**config['env_grid_search'], 'algorithm': config['algorithm']}
        for key, value in config['metrics'].items():
            if not isinstance(value, list):
                row[key] = value
        data[idx] = row
    return pd.DataFrame.from_dict(data, orient='index')


def normalize_value(
    value: float,
    min_val: float,
    max_val: float,
    higher_is_better: bool,
    method: str = 'max'
) -> float:
    """
    Normalize a value to 0-100 scale where 100 = best.

    Args:
        value: Original value
        min_val: Minimum value in the dataset
        max_val: Maximum value in the dataset
        higher_is_better: If True, higher values are better
        method: 'minmax' for (val-min)/(max-min), 'max' for val/max

    Returns:
        Normalized value (0-100)
    """
    # Handle edge case where max_val == 0 or max_val == min_val
    if max_val == 0 or max_val == min_val:
        # Return 50 (neutral) if value is at the min/max boundary, otherwise scale proportionally
        if value == min_val:
            return 50.0 if min_val != max_val else 50.0
        elif value == max_val and min_val == 0:
            return 50.0
        else:
            # Proportional scaling
            if max_val == 0:
                return 50.0
            return max(0, min(100, value / max_val * 100)) if max_val != min_val else 50.0

    if method == 'max':
        # Use value / max_value approach
        if higher_is_better:
            # Higher is better: normalize by max value directly
            normalized = value / max_val
        else:
            # Lower is better: (max - value) / (max - min) so min gets 100
            normalized = (max_val - value) / (max_val - min_val)
    else:
        # Traditional min-max normalization
        normalized = (value - min_val) / (max_val - min_val)
        if not higher_is_better:
            normalized = 1 - normalized

    # Ensure in 0-100 range and clip
    return max(0, min(100, normalized * 100))



def custom_palette():
    """Custom color palette."""
    q = list(sns.color_palette("deep"))
    q[1], q[9] = q[9], q[1]
    q[5], q[9] = q[9], q[5]
    q[6], q[9] = q[9], q[6]
    return q


def create_radar_chart(
    algorithms: List[str],
    metrics: List[str],
    values: Dict[str, Dict[str, float]],
    directions: Dict[str, bool],
    title: str,
    save_path: Path,
    figsize: Tuple[float, float] = (8, 6),
    font_size: int = 10,
    legend_font_size: int = 9,
    alpha: float = 0.15,
    normalize_method: str = 'max',
    rename_algo_label: Optional[Dict[str, str]] = None,
    rename_metric_label: Optional[Dict[str, str]] = None,
    metric_min_max: Optional[Dict[str, Tuple[float, float]]] = None,
    plt_style: str = 'seaborn-v0_8-colorblind',
    figure_face_color: str = '#FFFFFF',
    line_width: float = 1.5,
    marker_size: float = 5,
):
    """
    Create a radar/polygon chart for ablation comparison.
    """
    plt.style.use(plt_style)
    plt.rcParams['figure.facecolor'] = figure_face_color

    n_metrics = len(metrics)
    n_algorithms = len(algorithms)

    # Compute angles for each axis
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Apply metric label renaming
    display_metrics = [rename_metric_label.get(m, m) if rename_metric_label else m for m in metrics]

    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_metrics, size=font_size, fontweight='normal')

    # If metric_min_max not provided, compute from values
    if metric_min_max is None:
        metric_min_max = {m: (float('inf'), float('-inf')) for m in metrics}
        for algo in algorithms:
            for m in metrics:
                if m in values.get(algo, {}):
                    v = values[algo][m]
                    min_v, max_v = metric_min_max[m]
                    metric_min_max[m] = (min(min_v, v), max(max_v, v))

    # Colors for algorithms - use custom palette
    colors = custom_palette()[:n_algorithms]

    # Plot each algorithm
    for i, algo in enumerate(algorithms):
        # Get normalized values for this algorithm
        normalized_values = []
        for m in metrics:
            raw_val = values[algo][m]
            min_v, max_v = metric_min_max[m]
            higher_is_better = directions.get(m, True)
            norm_val = normalize_value(raw_val, min_v, max_v, higher_is_better, normalize_method)
            normalized_values.append(norm_val)

        # Close the polygon
        normalized_values += normalized_values[:1]

        # Apply algorithm label renaming
        display_algo = rename_algo_label.get(algo, algo) if rename_algo_label else algo

        # Plot with fill
        ax.fill(angles, normalized_values, alpha=alpha, color=colors[i])
        ax.plot(angles, normalized_values, 'o-', linewidth=line_width, color=colors[i],
                label=display_algo, markersize=marker_size)

    # Set radial limits and grid
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], size=font_size - 2, color='gray')
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)

    # Add title
    plt.title(title, size=font_size + 2, pad=20, fontweight='bold')

    # Add legend with centered, bold title
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05),
                       fontsize=legend_font_size, frameon=True)
    if legend:
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_ha('center')
        legend.get_frame().set_facecolor(figure_face_color)
        legend.get_frame().set_edgecolor('gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_ablation(
    data_dir: str = "charge_data_LMAPF_abalation",
    environments: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    higher_is_better: Optional[Dict[str, bool]] = None,
    hue_order: Optional[List[str]] = None,
    rename_algo_label: Optional[Dict[str, str]] = None,
    rename_metric_label: Optional[Dict[str, str]] = None,
    save_dir: str = ".",
    figsize: Tuple[float, float] = (8, 6),
    normalize_method: str = 'max',
    plt_style: str = 'seaborn-v0_8-colorblind',
    figure_face_color: str = '#FFFFFF',
):
    """
    Plot ablation experiment results.

    Args:
        data_dir: Directory containing ablation data
        environments: List of environment folders to aggregate (e.g., ['01-random'])
        metrics: List of metrics to plot.
        higher_is_better: Dict mapping metric -> True if higher is better.
                         MUST be explicitly defined for all metrics.
        hue_order: Order of algorithms for the legend
        rename_algo_label: Dict mapping algorithm name -> display name
        rename_metric_label: Dict mapping metric name -> display name
        save_dir: Directory to save plots
        figsize: Figure size
        normalize_method: 'max' (val/max*100) or 'minmax' ((val-min)/(max-min)*100)
        plt_style: Matplotlib style
        figure_face_color: Background color
    """
    # Default metrics
    default_metrics = [
        'avg_throughput',
        'valid_episode_relative',
        'agents_depletion_rate',
        'runtime',
    ]

    if metrics is None:
        metrics = default_metrics

    # Validate that all metrics have explicit higher_is_better defined
    if higher_is_better is None:
        raise ValueError("higher_is_better must be explicitly defined for all metrics")

    missing_metrics = [m for m in metrics if m not in higher_is_better]
    if missing_metrics:
        raise ValueError(f"higher_is_better not defined for metrics: {missing_metrics}")

    data_path = Path(data_dir)

    if environments is None:
        environments = sorted([d.name for d in data_path.iterdir() if d.is_dir()])

    print(f"Environments: {environments}")
    print(f"Metrics: {metrics}")
    print(f"Directions: {higher_is_better}")

    # Aggregate data across environments
    all_data = []
    for env in environments:
        env_path = data_path / env
        if not env_path.exists():
            continue

        yaml_files = list(env_path.glob('*.yaml'))
        yaml_config = None
        if yaml_files:
            with open(yaml_files[0], 'r') as f:
                yaml_config = yaml.safe_load(f)

        results = load_json_results(env_path)
        df = extract_metrics(results)
        df['environment'] = env
        all_data.append(df)

    if not all_data:
        print("No data found!")
        return

    df_combined = pd.concat(all_data, ignore_index=True)

    # Get algorithms
    available_algos = set(df_combined['algorithm'].unique())
    if hue_order is None:
        algorithms = sorted(available_algos)
    else:
        # Filter hue_order to only include algorithms that have data
        algorithms = [a for a in hue_order if a in available_algos]

    print(f"Algorithms: {algorithms}")

    # Calculate mean values per algorithm
    algo_values = {}
    for algo in algorithms:
        df_algo = df_combined[df_combined['algorithm'] == algo]
        algo_values[algo] = {}
        for m in metrics:
            if m in df_algo.columns:
                algo_values[algo][m] = df_algo[m].mean()
            else:
                algo_values[algo][m] = 0

    # Compute global metric_min_max from RAW DATA (not algorithm means)
    global_metric_min_max = {}
    for m in metrics:
        if m in df_combined.columns:
            global_metric_min_max[m] = (df_combined[m].min(), df_combined[m].max())
        else:
            global_metric_min_max[m] = (0, 1)

    # Create save directory
    save_path_dir = Path(save_dir)
    save_path_dir.mkdir(parents=True, exist_ok=True)

    # Plot for each environment combined
    title = f"Ablation Study - {', '.join(environments)}"
    save_file = save_path_dir / f"ablation_radar.pdf"

    create_radar_chart(
        algorithms=algorithms,
        metrics=metrics,
        values=algo_values,
        directions=higher_is_better,
        title=title,
        save_path=save_file,
        figsize=figsize,
        normalize_method=normalize_method,
        rename_algo_label=rename_algo_label,
        rename_metric_label=rename_metric_label,
        metric_min_max=global_metric_min_max,
        plt_style=plt_style,
        figure_face_color=figure_face_color,
    )

    # Also create individual environment plots
    for env in environments:
        env_path = data_path / env
        if not env_path.exists():
            continue

        results = load_json_results(env_path)
        df_env = extract_metrics(results)

        env_algo_values = {}
        for algo in algorithms:
            df_algo = df_env[df_env['algorithm'] == algo]
            env_algo_values[algo] = {}
            for m in metrics:
                if m in df_algo.columns:
                    env_algo_values[algo][m] = df_algo[m].mean()
                else:
                    env_algo_values[algo][m] = 0

        # Compute per-environment global metric_min_max from raw data
        env_metric_min_max = {}
        for m in metrics:
            if m in df_env.columns:
                env_metric_min_max[m] = (df_env[m].min(), df_env[m].max())
            else:
                env_metric_min_max[m] = (0, 1)

        save_file = save_path_dir / f"ablation_{env}_radar.pdf"

        create_radar_chart(
            algorithms=algorithms,
            metrics=metrics,
            values=env_algo_values,
            directions=higher_is_better,
            title=f"Ablation Study - {env}",
            save_path=save_file,
            figsize=figsize,
            normalize_method=normalize_method,
            rename_algo_label=rename_algo_label,
            rename_metric_label=rename_metric_label,
            metric_min_max=env_metric_min_max,
            plt_style=plt_style,
            figure_face_color=figure_face_color,
        )

    print(f"\nDone! Plots saved to {save_dir}")


if __name__ == "__main__":
    plot_ablation(
        data_dir="charge_data_LMAPF_abalation",
        environments=None,  # All environments
        metrics=[
            'avg_throughput',
            'valid_episode_relative',
            'agents_depletion_rate',
            'avg_relative_battery',
            'avg_charging_rate',
            'avg_goal_battery_relative',
            #'runtime'
        ],
        higher_is_better={
            'avg_throughput': True,
            'valid_episode_relative': True,
            'agents_depletion_rate': False,  # Lower is better
            'avg_relative_battery': True,
            'avg_charging_rate': True,
            'avg_goal_battery_relative': True,
            #'runtime': False,
        },
        hue_order=['ChargerAppo(full)', 'ChargerAppo(wo_enc)', 'ChargerAppo(wo_battery)', 'ChargerAppo(wo_penalty)', 'ChargerAppo(new_obs)'],
        rename_algo_label={
            'ChargerAppo(full)': 'Full',
            'ChargerAppo(new_obs)': 'w/o position', 
            'ChargerAppo(wo_enc)': 'w/o enc',
            'ChargerAppo(wo_battery)': 'w/o battery',
            'ChargerAppo(wo_penalty)': 'w/o penalty',
        },
        rename_metric_label={
            'avg_throughput': 'Throughput',
            'valid_episode_relative': 'Episode',
            'agents_depletion_rate': 'Depletion',  # Lower is better
            'avg_relative_battery': 'Battery',
            'avg_charging_rate': 'Charging',
            'avg_goal_battery_relative': 'Target',
            #'runtime': 'Runtime'
        },
        save_dir="ablation_plots",
        figsize=(6, 8),
        normalize_method='max',  # 'max' for val/max, 'minmax' for traditional
    )
