"""
Compare algorithm performance between pogema (raw_data_LMAPF) and
pogema-charge (charge_data_LMAPF) environments using line plots.
"""

import json
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import yaml


def load_json_results(folder_path: Path) -> List[dict]:
    """Load all JSON result files from a folder."""
    results = []
    for file in folder_path.glob('*.json'):
        with open(file, 'r') as f:
            results.extend(json.load(f))
    return results


def extract_metrics(results: List[dict], env_name: str) -> pd.DataFrame:
    """Convert results list to DataFrame with environment label."""
    data = {}
    for idx, config in enumerate(results):
        row = {**config['env_grid_search'], 'algorithm': config['algorithm'], 'environment': env_name}
        for key, value in config['metrics'].items():
            if not isinstance(value, list):
                row[key] = value
        data[idx] = row
    df = pd.DataFrame.from_dict(data, orient='index')
    return df


def load_yaml_config(folder_path: Path) -> Optional[Dict]:
    """Load the yaml config file from folder."""
    yaml_files = list(folder_path.glob('*.yaml'))
    if not yaml_files:
        return None
    with open(yaml_files[0], 'r') as f:
        return yaml.safe_load(f)


def custom_palette():
    q = list(sns.color_palette("deep"))
    q[1], q[9] = q[9], q[1]
    q[5], q[9] = q[9], q[5]
    q[6], q[9] = q[9], q[6]
    return q


def plot_comparison_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    style: str,
    title: str,
    save_path: Path,
    # Ordering
    hue_order: Optional[List[str]] = None,
    style_order: Optional[List[str]] = None,
    # Axis
    ticks: Optional[List[int]] = None,
    use_log_scale_x: bool = True,
    use_log_scale_y: bool = False,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    # Figure
    width: float = 3.5,
    height: float = 2.6,
    figure_face_color: str = '#FFFFFF',
    # Text
    font_size: int = 8,
    legend_font_size: int = 8,
    # Lines & Markers
    line_width: float = 2.0,
    markers: bool = True,
    error_bar: str = 'ci',  # 'ci', 'sd', 'se', or None
    error_bar_n: int = 95,
    # Style
    plt_style: str = 'seaborn-v0_8-colorblind',
    palette: Optional[List[str]] = None,
    # Legend
    legend_loc: str = 'best',
    remove_legend_title: bool = True,
    # Output
    extension: str = 'pdf',
):
    """Create a line plot comparison using seaborn."""
    plt.style.use(plt_style)
    plt.rcParams['figure.figsize'] = (width, height)
    plt.rcParams['font.size'] = font_size
    plt.rcParams['legend.fontsize'] = legend_font_size
    plt.rcParams['figure.facecolor'] = figure_face_color

    fig, ax = plt.subplots()

    if hue_order is None:
        hue_order = sorted(df[hue].unique())
    if style_order is None:
        style_order = sorted(df[style].unique())

    # Filter to only rows with valid values
    df_plot = df[[x, y, hue, style]].dropna()

    # Build errorbar tuple
    if error_bar is None:
        errorbar = None
    elif error_bar == 'ci':
        errorbar = (error_bar, error_bar_n)
    else:
        errorbar = (error_bar,)

    # Use custom palette or default
    if palette is None:
        palette = custom_palette()[:len(hue_order)]

    sns.lineplot(
        x=x, y=y, data=df_plot, errorbar=errorbar,
        hue=hue, hue_order=hue_order,
        style=style, style_order=style_order,
        linewidth=line_width, markers=markers,
        palette=palette,
        ax=ax
    )

    ax.set_title(title)

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    # Set legend with centered, bold title
    legend = ax.legend(loc=legend_loc)
    if legend:
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_ha('center')
        if remove_legend_title:
            legend.set_title('')

    if use_log_scale_x:
        ax.set_xscale('log', base=2)
        from matplotlib.ticker import ScalarFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter())

    if use_log_scale_y:
        ax.set_yscale('log', base=2)
        from matplotlib.ticker import ScalarFormatter
        ax.yaxis.set_major_formatter(ScalarFormatter())

    if ticks:
        ax.set_xticks(np.array(ticks))

    plt.tight_layout()
    plt.grid()
    plt.savefig(save_path.with_suffix(f'.{extension}'))
    print(f"  Saved: {save_path.with_suffix(f'.{extension}')}")
    plt.close()


def compare_environments(
    charge_data_dir: str = "charge_data_LMAPF",
    raw_data_dir: str = "raw_data_LMAPF",
    environments: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    save_dir: str = ".",
    plot_config: Optional[Dict] = None,
):
    """
    Compare algorithm performance between pogema and pogema-charge environments.

    Args:
        charge_data_dir: Path to charge environment results
        raw_data_dir: Path to raw/pogema environment results
        environments: List of environment subdirectories to compare (e.g., ['01-random'])
        metrics: List of metrics to plot ['avg_throughput', 'valid_episode_relative']
        save_dir: Directory to save plots
        plot_config: Override config for plotting. Keys:
            - width, height: figure size
            - font_size, legend_font_size: text sizes
            - line_width: line thickness
            - markers: show markers (bool)
            - error_bar: 'ci', 'sd', 'se', or None
            - error_bar_n: confidence interval percentage (default 95)
            - use_log_scale_x, use_log_scale_y: log scale
            - x_label, y_label: axis labels
            - legend_loc: legend position
            - remove_legend_title: hide legend title
            - plt_style: matplotlib style
            - extension: 'pdf', 'png', 'svg', etc.
    """
    if metrics is None:
        metrics = ['avg_throughput', 'valid_episode_relative']

    charge_path = Path(charge_data_dir)
    raw_path = Path(raw_data_dir)

    # Create save directory if it doesn't exist
    save_path_dir = Path(save_dir)
    save_path_dir.mkdir(parents=True, exist_ok=True)

    if environments is None:
        charge_envs = set(d.name for d in charge_path.iterdir() if d.is_dir())
        raw_envs = set(d.name for d in raw_path.iterdir() if d.is_dir())
        environments = sorted(charge_envs & raw_envs)

    print(f"Environments: {environments}")

    # Load and process data for each environment
    for env in environments:
        charge_env_path = charge_path / env
        raw_env_path = raw_path / env

        if not charge_env_path.exists() and not raw_env_path.exists():
            continue

        print(f"\n=== Processing {env} ===")

        # Load yaml config for this environment
        charge_yaml = load_yaml_config(charge_env_path)
        raw_yaml = load_yaml_config(raw_env_path)

        # Use charge yaml's results_views for plot configuration
        if charge_yaml and 'results_views' in charge_yaml:
            views_config = charge_yaml['results_views']
        else:
            views_config = {}

        charge_results = load_json_results(charge_env_path) if charge_env_path.exists() else []
        raw_results = load_json_results(raw_env_path) if raw_env_path.exists() else []

        df_charge = extract_metrics(charge_results, 'charge')
        df_raw = extract_metrics(raw_results, 'pogema')

        # Set valid_episode_relative = 1 for pogema environment
        if 'valid_episode_relative' not in df_raw.columns:
            df_raw['valid_episode_relative'] = 1.0
        else:
            df_raw['valid_episode_relative'] = 1.0

        # Combine dataframes - 'environment' will be used as style
        df_combined = pd.concat([df_charge, df_raw], ignore_index=True)

        # Get all unique algorithms across both environments
        all_algos = sorted(set(df_charge['algorithm'].unique()) | set(df_raw['algorithm'].unique()))
        print(f"  Algorithms: {all_algos}")

        for metric in metrics:
            # Find matching view in yaml or use defaults
            view_cfg = None
            for view_name, view_data in views_config.items():
                if view_data.get('type') == 'plot' and view_data.get('y') == metric:
                    view_cfg = view_data
                    break

            if view_cfg:
                x_axis = view_cfg.get('x', 'num_agents')
                title = view_cfg.get('name', f'{env} - {metric}')
                ticks = view_cfg.get('ticks')
                width = view_cfg.get('width', 3.5)
                height = view_cfg.get('height', 2.6)
                font_size = view_cfg.get('font_size', 8)
                legend_font_size = view_cfg.get('legend_font_size', 8)
                line_width = view_cfg.get('line_width', 2.0)
                use_log_scale_x = view_cfg.get('use_log_scale_x', True)
                use_log_scale_y = view_cfg.get('use_log_scale_y', False)
                hue_order = view_cfg.get('hue_order')
                markers = view_cfg.get('markers', True)
                error_bar = view_cfg.get('error_bar', 'ci')
                error_bar_n = view_cfg.get('error_bar_n', 95)
                legend_loc = view_cfg.get('legend_loc', 'best')
                remove_legend_title = view_cfg.get('remove_legend_title', True)
                x_label = view_cfg.get('x_label')
                y_label = view_cfg.get('y_label')
            else:
                # Default configuration
                x_axis = 'num_agents'
                title = f'{env} - {metric}'
                ticks = [8, 16, 24, 32, 48, 64]
                width, height = 3.5, 2.6
                font_size, legend_font_size = 8, 8
                line_width, use_log_scale_x = 2.0, True
                use_log_scale_y = False
                hue_order = None
                markers = True
                error_bar = 'ci'
                error_bar_n = 95
                legend_loc = 'best'
                remove_legend_title = True
                x_label = None
                y_label = None

            # Apply plot_config overrides
            if plot_config:
                width = plot_config.get('width', width)
                height = plot_config.get('height', height)
                font_size = plot_config.get('font_size', font_size)
                legend_font_size = plot_config.get('legend_font_size', legend_font_size)
                line_width = plot_config.get('line_width', line_width)
                markers = plot_config.get('markers', markers)
                error_bar = plot_config.get('error_bar', error_bar)
                error_bar_n = plot_config.get('error_bar_n', error_bar_n)
                use_log_scale_x = plot_config.get('use_log_scale_x', use_log_scale_x)
                use_log_scale_y = plot_config.get('use_log_scale_y', use_log_scale_y)
                legend_loc = plot_config.get('legend_loc', legend_loc)
                remove_legend_title = plot_config.get('remove_legend_title', remove_legend_title)
                x_label = plot_config.get('x_label', x_label)
                y_label = plot_config.get('y_label', y_label)
                if 'hue_order' in plot_config:
                    hue_order = plot_config['hue_order']
                if 'ticks' in plot_config:
                    ticks = plot_config['ticks']

            print(f"  Metric: {metric}, x={x_axis}, view='{view_cfg.get('name') if view_cfg else 'default'}'")

            # Check if metric exists in combined data
            if metric not in df_combined.columns:
                print(f"    Skipping - {metric} not in data")
                continue

            # Filter data for this metric
            df_metric = df_combined[df_combined[metric].notna()].copy()

            if len(df_metric) == 0:
                print(f"    Skipping - no data after filtering")
                continue

            # Determine hue_order - algorithms should be ordered
            if hue_order is None:
                # Put charge algorithms first, then pogema, both filtered by presence
                charge_algos = [a for a in all_algos if a in df_charge['algorithm'].unique()]
                pogema_algos = [a for a in all_algos if a in df_raw['algorithm'].unique()]
                hue_order = charge_algos + pogema_algos
            else:
                # Filter hue_order to only include algorithms present in data
                hue_order = [a for a in hue_order if a in all_algos]

            # Style order: charge first, then pogema
            style_order = ['charge', 'pogema']

            save_path = Path(save_dir) / f'comparison_{env}_{metric}'

            plot_comparison_line(
                df=df_metric,
                x=x_axis,
                y=metric,
                hue='algorithm',      # Color by algorithm
                style='environment',  # Line type by environment (charge vs pogema)
                title=title,
                save_path=save_path,
                hue_order=hue_order,
                style_order=style_order,
                ticks=ticks,
                use_log_scale_x=use_log_scale_x,
                use_log_scale_y=use_log_scale_y,
                width=width,
                height=height,
                font_size=font_size,
                legend_font_size=legend_font_size,
                line_width=line_width,
                markers=markers,
                error_bar=error_bar,
                error_bar_n=error_bar_n,
                legend_loc=legend_loc,
                remove_legend_title=remove_legend_title,
                x_label=x_label,
                y_label=y_label,
            )

    print(f"\nDone! Plots saved to {save_dir}")


if __name__ == "__main__":
    compare_environments(
        charge_data_dir="charge_data_LMAPF_env",
        raw_data_dir="raw_data_LMAPF_env",
        environments=None,  # All environments
        metrics=['avg_throughput'],
        save_dir="compare_environment_throughput"
    )
