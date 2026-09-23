"""
Generate statistics for experimental settings in LaTeX table format.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any
import json


def load_yaml_config(folder_path: Path) -> dict:
    """Load the yaml config file from a folder."""
    yaml_files = list(folder_path.glob('*.yaml'))
    if yaml_files:
        with open(yaml_files[0], 'r') as f:
            return yaml.safe_load(f)
    return {}


def extract_config_info(config: dict, env_name: str) -> Dict[str, Any]:
    """Extract relevant configuration information."""
    env_config = config.get('environment', {})
    algo_config = config.get('algorithms', {})

    # Count maps
    map_name = env_config.get('map_name', {})
    if isinstance(map_name, dict):
        maps = map_name.get('grid_search', [])
    elif isinstance(map_name, list):
        maps = map_name
    else:
        maps = [map_name] if map_name else []

    # Extract num_agents
    num_agents = env_config.get('num_agents', {})
    if isinstance(num_agents, dict):
        num_agents_list = num_agents.get('grid_search', [])
    elif isinstance(num_agents, list):
        num_agents_list = num_agents
    else:
        num_agents_list = [num_agents] if num_agents else []

    # Extract max_episode_steps
    max_episode_steps = env_config.get('max_episode_steps', {})
    if isinstance(max_episode_steps, dict):
        max_episode_steps_list = max_episode_steps.get('grid_search', [])
    elif isinstance(max_episode_steps, list):
        max_episode_steps_list = max_episode_steps
    else:
        max_episode_steps_list = [max_episode_steps] if max_episode_steps else []

    # Extract seeds
    seed = env_config.get('seed', {})
    if isinstance(seed, dict):
        seed_list = seed.get('grid_search', [])
    elif isinstance(seed, list):
        seed_list = seed
    else:
        seed_list = [seed] if seed else []

    # Calculate total experiments
    total_maps = len(maps)
    total_agents = len(num_agents_list)
    total_steps = len(max_episode_steps_list)
    total_seeds = len(seed_list) if seed_list else 1

    total_experiments = total_maps * total_agents * total_steps * total_seeds

    # Get observation type and agent per charge
    observation_type = env_config.get('observation_type', 'N/A')
    agent_per_charge = env_config.get('agent_per_charge', 'N/A')

    # Get algorithm names
    algorithms = list(algo_config.keys()) if algo_config else []

    # Extract algorithm parameters
    algo_params = {}
    for algo_name, algo_params_dict in algo_config.items():
        params = {k: v for k, v in algo_params_dict.items() if k != 'name'}
        algo_params[algo_name] = params

    return {
        'num_maps': total_maps,
        'map_names': maps,
        'num_agents': num_agents_list,
        'max_episode_steps': max_episode_steps_list,
        'seeds': seed_list,
        'total_experiments': total_experiments,
        'observation_type': observation_type,
        'agent_per_charge': agent_per_charge,
        'algorithms': algorithms,
        'algorithm_params': algo_params,
    }


def generate_settings_latex_table(
    data_dir: str = "charge_data_LMAPF",
    environments: Dict[str, str] = None,
    output_file: str = "settings_table.txt",
):
    """
    Generate LaTeX table with experimental settings.
    """
    if environments is None:
        environments = {
            'Maze': '02-mazes',
            'Warehouse': '03-warehouse',
            'MovingAI': '04-movingai',
        }

    results = {}
    for display_name, folder_name in environments.items():
        env_path = Path(data_dir) / folder_name
        if not env_path.exists():
            print(f"Warning: {env_path} does not exist, skipping...")
            continue

        config = load_yaml_config(env_path)
        info = extract_config_info(config, display_name)
        results[display_name] = info

        print(f"\n{display_name} ({folder_name}):")
        print(f"  Maps: {info['num_maps']}")
        print(f"  num_agents: {info['num_agents']}")
        print(f"  max_episode_steps: {info['max_episode_steps']}")
        print(f"  Seeds: {info['seeds']}")
        print(f"  Total experiments: {info['total_experiments']}")
        print(f"  Observation type: {info['observation_type']}")
        print(f"  Agent per charge: {info['agent_per_charge']}")
        print(f"  Algorithms: {info['algorithms']}")

    # Generate LaTeX table
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("% Experimental settings table for LaTeX\n\n")

        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{实验环境配置统计}\n")
        f.write("\\label{tab:experimental_settings}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("环境 & 地图数量 & 智能体数量 & 最大步数 \\\\\n")
        f.write("\\midrule\n")

        for env_name, folder_name in environments.items():
            if env_name in results:
                info = results[env_name]
                maps_str = str(info['num_maps'])
                agents_str = str(info['num_agents'])
                steps_str = str(info['max_episode_steps'])
                f.write(f"{env_name} & {maps_str} & {agents_str} & {steps_str} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")

        # Second table: detailed settings
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{实验参数详细设置}\n")
        f.write("\\label{tab:experimental_details}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("环境 & 观测类型 & 每充电桩服务智能体数 & 算法数量 & 总实验次数 & 地图名称 & 随机种子 \\\\\n")
        f.write("\\midrule\n")

        for env_name, folder_name in environments.items():
            if env_name in results:
                info = results[env_name]
                obs_type = info['observation_type']
                apc = info['agent_per_charge']
                num_algos = len(info['algorithms'])
                total_exp = info['total_experiments']
                seeds = str(info['seeds']) if info['seeds'] else '0'

                # Truncate seeds if too long
                if len(seeds) > 30:
                    seeds = f"{{{len(info['seeds'])} seeds}}"

                f.write(f"{env_name} & {obs_type} & {apc} & {num_algos} & {total_exp} & {len(info['map_names'])} maps & {seeds} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")

        # Third table: algorithm parameters
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{算法参数配置}\n")
        f.write("\\label{tab:algorithm_params}\n")

        # Collect all unique algorithms
        all_algos = set()
        for info in results.values():
            all_algos.update(info['algorithms'])

        algo_list = sorted(list(all_algos))

        # Create column format
        col_format = 'l' + 'c' * len(environments)
        f.write(f"\\begin{{tabular}}{{{col_format}}}\n")
        f.write("\\toprule\n")

        # Header
        env_names = " & ".join(environments.keys())
        f.write(f"算法 & {env_names} \\\\\n")
        f.write("\\midrule\n")

        for algo in algo_list:
            row = [algo]
            for env_name in environments.keys():
                if env_name in results:
                    info = results[env_name]
                    if algo in info['algorithm_params']:
                        params = info['algorithm_params'][algo]
                        # Format key params
                        param_strs = []
                        for k, v in params.items():
                            if k == 'num_process':
                                param_strs.append(f"proc={v}")
                            elif k == 'time_limit':
                                param_strs.append(f"tlim={v}")
                            elif k == 'planning_window':
                                param_strs.append(f"pw={v}")
                            elif k == 'simulation_window':
                                param_strs.append(f"sw={v}")
                            elif k == 'solver':
                                param_strs.append(f"solver={v}")
                            elif k == 'low_level_planner':
                                param_strs.append(f"llp={v}")
                            elif k == 'parallel_backend':
                                param_strs.append(f"pb={v}")
                        row.append(", ".join(param_strs) if param_strs else '✓')
                    else:
                        row.append('-')
                else:
                    row.append('-')
            f.write(" & ".join(row) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")

        # Fourth table: map names for each environment
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{地图配置}\n")
        f.write("\\label{tab:map_configs}\n")

        for env_name, folder_name in environments.items():
            if env_name in results:
                info = results[env_name]
                maps = info['map_names']

                f.write(f"\\paragraph{{{env_name} ({len(maps)} maps):}}\n")
                f.write("\\begin{itemize}\n")

                if len(maps) <= 10:
                    for m in maps:
                        f.write(f"\\item {m}\n")
                else:
                    # Group maps by prefix
                    map_groups = {}
                    for m in maps:
                        prefix = m.rsplit('-', 1)[0] if '-' in m else m
                        if prefix not in map_groups:
                            map_groups[prefix] = []
                        map_groups[prefix].append(m)

                    for prefix, group_maps in map_groups.items():
                        count = len(group_maps)
                        sample = group_maps[0] if group_maps else ""
                        if count == 1:
                            f.write(f"\\item {sample}\n")
                        else:
                            f.write(f"\\item {prefix}-*: {count} maps (e.g., {sample})\n")

                f.write("\\end{itemize}\n\n")

    print(f"\nSaved LaTeX tables to {output_file}")

    # Also save as JSON for reference
    json_output = output_file.replace('.txt', '.json')
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON to {json_output}")

    return results


if __name__ == "__main__":
    generate_settings_latex_table(
        data_dir="charge_data_LMAPF",
        environments={
            'Maze': '02-mazes',
            'Warehouse': '03-warehouse',
            'MovingAI': '04-movingai',
        },
        output_file="settings_table.txt",
    )