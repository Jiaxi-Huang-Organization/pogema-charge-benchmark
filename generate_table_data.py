"""
Generate statistics for LaTeX tables from charge_data_LMAPF.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


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


def calculate_statistics(df: pd.DataFrame, metrics: List[str]) -> Dict[str, Dict[str, float]]:
    """Calculate mean and std for each algorithm across all data points."""
    stats = {}
    for algo in df['algorithm'].unique():
        df_algo = df[df['algorithm'] == algo]
        stats[algo] = {}
        for m in metrics:
            if m in df_algo.columns:
                values = df_algo[m].dropna()
                if len(values) > 0:
                    stats[algo][m] = {
                        'mean': values.mean(),
                        'std': values.std(),
                    }
    return stats


def generate_table_data(
    data_dir: str = "charge_data_LMAPF",
    environments: Dict[str, str] = None,
    metrics: Dict[str, str] = None,
    output_file: str = "table_data.json",
    generate_runtime_table: bool = True,
):
    """
    Generate statistics for LaTeX tables.

    Args:
        data_dir: Directory containing charge_data_LMAPF
        environments: Dict mapping display name -> folder name
        metrics: Dict mapping display name -> metric name
        output_file: Output file path
    """
    if environments is None:
        environments = {
            'Maze': '02-mazes',
            'Warehouse': '03-warehouse',
            'MovingAI': '04-movingai',
        }

    if metrics is None:
        metrics = {
            '吞吐量': 'avg_throughput',
            '充电率': 'avg_relative_battery',
            '存活率': 'valid_episode_relative',
        }

    results = {}

    for display_name, folder_name in environments.items():
        env_path = Path(data_dir) / folder_name
        if not env_path.exists():
            print(f"Warning: {env_path} does not exist, skipping...")
            continue

        json_results = load_json_results(env_path)
        df = extract_metrics(json_results)

        stats = calculate_statistics(df, list(metrics.values()))
        results[display_name] = stats

        print(f"\n{display_name} ({folder_name}):")
        for algo, algo_stats in sorted(stats.items()):
            print(f"  {algo}:")
            for disp_name, metric_name in metrics.items():
                if metric_name in algo_stats:
                    m = algo_stats[metric_name]
                    print(f"    {disp_name}: {m['mean']:.4f} ± {m['std']:.4f}")

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_file}")

    # Also generate LaTeX table format
    latex_file = output_file.replace('.json', '.txt')
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write("% Table data for LaTeX\n\n")

        # Get all algorithms that appear in any environment
        all_algos = set()
        for env_stats in results.values():
            all_algos.update(env_stats.keys())

        # Filter to include only relevant algorithms for the table
        table_algos = ['RHCR', 'DCC', 'SCRIMP', 'Follower', 'ChargerAppo']
        table_algos = [a for a in table_algos if a in all_algos]

        # Header
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{各算法性能对比}\n")
        f.write("\\label{tab:algorithm_comparison}\n")

        # Column format
        num_envs = len(environments)
        num_metrics = len(metrics)
        col_format = 'l' + 'c' * (num_envs * num_metrics)
        f.write(f"\\begin{{tabular}}{{{col_format}}}\n")
        f.write("\\toprule\n")

        # Environment headers
        env_headers = " & ".join([f"\\multicolumn{{{num_metrics}}}{{c}}{{{env}}}" for env in environments.keys()])
        f.write(f" & {env_headers} \\\\\n")

        # Metric sub-headers
        metric_names = " & ".join([f"\\multicolumn{1}{{c}}{{{m}}}" for m in metrics.keys()])
        metric_row = " & ".join([metric_names] * len(environments))
        f.write(f"\\cmidrule{{(lr){{2-{1 + num_envs * num_metrics}}}}}\n")
        f.write(f"算法 & {metric_row} \\\\\n")
        f.write("\\midrule\n")

        # Data rows
        for algo in table_algos:
            row = [algo]
            for env_name, folder_name in environments.items():
                if env_name in results and algo in results[env_name]:
                    algo_stats = results[env_name][algo]
                    for disp_name, metric_name in metrics.items():
                        if metric_name in algo_stats:
                            m = algo_stats[metric_name]
                            row.append(f"${m['mean']:.3f}$")
                        else:
                            row.append("-")
                else:
                    row.extend(["-"] * num_metrics)
            f.write(" & ".join(row) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved LaTeX format to {latex_file}")

    # Generate runtime table
    runtime_metrics = {
        '运行时间': 'runtime',
    }

    runtime_results = {}
    for display_name, folder_name in environments.items():
        env_path = Path(data_dir) / folder_name
        if not env_path.exists():
            continue

        json_results = load_json_results(env_path)
        df = extract_metrics(json_results)

        stats = calculate_statistics(df, list(runtime_metrics.values()))
        runtime_results[display_name] = stats

        print(f"\n{display_name} Runtime:")
        for algo, algo_stats in sorted(stats.items()):
            print(f"  {algo}:")
            for disp_name, metric_name in runtime_metrics.items():
                if metric_name in algo_stats:
                    m = algo_stats[metric_name]
                    print(f"    {disp_name}: {m['mean']:.4f} ± {m['std']:.4f}")

    # Calculate average runtime across environments for each algorithm
    algo_avg_runtime = {}
    for algo in all_algos:
        runtime_sum = 0
        runtime_count = 0
        for env_name in runtime_results:
            if algo in runtime_results[env_name] and 'runtime' in runtime_results[env_name][algo]:
                runtime_sum += runtime_results[env_name][algo]['runtime']['mean']
                runtime_count += 1
        if runtime_count > 0:
            algo_avg_runtime[algo] = runtime_sum / runtime_count

    # Save runtime LaTeX table
    runtime_latex_file = output_file.replace('.json', '_runtime.txt')
    with open(runtime_latex_file, 'w', encoding='utf-8') as f:
        f.write("% Runtime table data for LaTeX\n\n")

        all_algos = set()
        for env_stats in runtime_results.values():
            all_algos.update(env_stats.keys())

        table_algos = ['RHCR', 'DCC', 'SCRIMP', 'Follower', 'ChargerAppo']
        table_algos = [a for a in table_algos if a in all_algos]

        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{各算法运行时间对比}\n")
        f.write("\\label{tab:runtime_comparison}\n")

        num_envs = len(environments)
        num_metrics = len(runtime_metrics)
        col_format = 'l' + 'c' * (num_envs * num_metrics) + 'c'
        f.write(f"\\begin{{tabular}}{{{col_format}}}\n")
        f.write("\\toprule\n")

        env_headers = " & ".join([f"\\multicolumn{{{num_metrics}}}{{c}}{{{env}}}" for env in environments.keys()])
        f.write(f" & {env_headers} & {{}} \\\\\n")

        metric_names = " & ".join([f"\\multicolumn{1}{{c}}{{{m}}}" for m in runtime_metrics.keys()])
        metric_row = " & ".join([metric_names] * len(environments))
        f.write(f"\\cmidrule{{(lr){{2-{1 + num_envs * num_metrics + 1}}}}}\n")
        f.write(f"算法 & {metric_row} & {{平均}} \\\\\n")
        f.write("\\midrule\n")

        for algo in table_algos:
            row = [algo]
            for env_name, folder_name in environments.items():
                if env_name in runtime_results and algo in runtime_results[env_name]:
                    algo_stats = runtime_results[env_name][algo]
                    for disp_name, metric_name in runtime_metrics.items():
                        if metric_name in algo_stats:
                            m = algo_stats[metric_name]
                            row.append(f"${m['mean']:.3f}$")
                        else:
                            row.append("-")
                else:
                    row.extend(["-"] * num_metrics)
            # Add average
            if algo in algo_avg_runtime:
                row.append(f"${algo_avg_runtime[algo]:.3f}$")
            else:
                row.append("-")
            f.write(" & ".join(row) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved runtime LaTeX format to {runtime_latex_file}")

    return results


if __name__ == "__main__":
    generate_table_data(
        data_dir="charge_data_LMAPF",
        environments={
            'Maze': '02-mazes',
            'Warehouse': '03-warehouse',
            'MovingAI': '04-movingai',
        },
        metrics={
            '吞吐量': 'avg_throughput',
            '充电率': 'avg_charging_rate',
            '存活率': 'valid_episode_relative',
        },
        output_file="table_data.json",
    )
