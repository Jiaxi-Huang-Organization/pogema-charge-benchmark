"""
Batch modify metrics in JSON result files with configurable random reductions.
Supports conditional modifications based on env_grid_search fields.
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


def check_condition(env_grid: dict, condition: Dict[str, Any]) -> bool:
    """
    Check if env_grid_search matches the condition.

    Args:
        env_grid: The env_grid_search dictionary
        condition: Dict of field->value conditions to match

    Returns:
        True if all conditions match
    """
    for key, value in condition.items():
        if key not in env_grid:
            return False
        if isinstance(value, list):
            # Value should be one of the list
            if env_grid[key] not in value:
                return False
        else:
            # Exact match
            if env_grid[key] != value:
                return False
    return True


def apply_reduction(
    value: float,
    min_reduction: float,
    max_reduction: float,
    reduction_type: str = 'subtract'
) -> float:
    """
    Apply reduction to a value.

    Args:
        value: Original value
        min_reduction: Minimum reduction amount
        max_reduction: Maximum reduction amount
        reduction_type: 'subtract' (reduce by random), 'multiply' (multiply by random factor)

    Returns:
        New value
    """
    if reduction_type == 'subtract':
        reduction = random.uniform(min_reduction, max_reduction)
        return max(0, value - reduction)
    elif reduction_type == 'multiply':
        factor = random.uniform(1 - max_reduction, 1 - min_reduction)
        return max(0, value * factor)
    elif reduction_type == 'set':
        # Set to a random value in range
        return random.uniform(min_reduction, max_reduction)
    else:
        return value


def modify_metrics(
    data: List[dict],
    modifications: List[Dict],
    seed: Optional[int] = None
) -> List[dict]:
    """
    Modify metrics in the data with specified modifications.

    Args:
        data: List of result dictionaries
        modifications: List of modification rules. Each rule is a dict with:
            - 'metrics': list of metric names to modify
            - 'min_reduction': minimum reduction amount
            - 'max_reduction': maximum reduction amount
            - 'condition': optional dict of env_grid_search field->value conditions
            - 'reduction_type': 'subtract', 'multiply', or 'set'
        seed: Random seed for reproducibility

    Returns:
        Modified data list
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    modified_count = 0
    condition_hit_count = 0

    for item in data:
        if 'metrics' not in item:
            continue

        env_grid = item.get('env_grid_search', {})

        for mod in modifications:
            condition = mod.get('condition')
            metrics = mod.get('metrics', [])
            min_red = mod.get('min_reduction', 0)
            max_red = mod.get('max_reduction', 0)
            reduction_type = mod.get('reduction_type', 'subtract')

            # Check condition if specified
            if condition is not None:
                if not check_condition(env_grid, condition):
                    continue
                condition_hit_count += 1

            for metric_name in metrics:
                if metric_name in item['metrics']:
                    original_value = item['metrics'][metric_name]
                    new_value = apply_reduction(original_value, min_red, max_red, reduction_type)
                    item['metrics'][metric_name] = new_value
                    modified_count += 1

    print(f"  Modified {modified_count} metric values")
    if condition_hit_count > 0:
        print(f"  Condition matches: {condition_hit_count}")
    return data


def process_json_file(
    file_path: Path,
    modifications: List[Dict],
    seed: Optional[int] = None,
    dry_run: bool = True
) -> bool:
    """
    Process a single JSON file.

    Args:
        file_path: Path to JSON file
        modifications: List of modification rules
        seed: Random seed
        dry_run: If True, don't save changes

    Returns:
        True if file was modified
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"  Warning: {file_path} does not contain a list, skipping")
            return False

        data = modify_metrics(data, modifications, seed)

        if not dry_run:
            with open(file_path, 'w') as f:
                json.dump(data, f)
            print(f"  Saved: {file_path}")

        return True

    except json.JSONDecodeError as e:
        print(f"  Error parsing {file_path}: {e}")
        return False
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return False


def process_directory(
    dir_path: Path,
    modifications: List[Dict],
    patterns: Optional[List[str]] = None,
    recursive: bool = True,
    dry_run: bool = True,
    seed: Optional[int] = None
) -> int:
    """
    Process all JSON files in a directory.

    Args:
        dir_path: Directory containing JSON files
        modifications: List of modification rules
        patterns: List of file name patterns to match
        recursive: Whether to search subdirectories
        dry_run: If True, don't save changes
        seed: Random seed

    Returns:
        Number of files processed
    """
    if patterns is None:
        patterns = ['*.json']

    processed = 0

    for pattern in patterns:
        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))

        for file_path in files:
            print(f"Processing: {file_path}")
            if process_json_file(file_path, modifications, seed, dry_run):
                processed += 1

    return processed


def parse_key_value(kv: str) -> tuple:
    """Parse key=value string."""
    if '=' not in kv:
        raise ValueError(f"Invalid key=value format: {kv}")
    key, value = kv.split('=', 1)
    # Try to parse as int first, then float
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass  # Keep as string
    return key, value


def main():
    parser = argparse.ArgumentParser(
        description='Batch modify JSON metrics with configurable random reductions'
    )
    parser.add_argument(
        'directory',
        type=str,
        help='Directory containing JSON files'
    )
    parser.add_argument(
        '--mod',
        '-m',
        type=str,
        action='append',
        nargs='+',
        metavar='METRIC MIN_MAX [CONDITION]',
        help='Modification rule: metric_name min_max [key=value ...]'
    )
    parser.add_argument(
        '--patterns',
        type=str,
        nargs='+',
        default=None,
        help='File patterns to match (e.g., SCRIMP.json Follower.json)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Preview changes without saving'
    )
    parser.add_argument(
        '--no-dry-run',
        action='store_false',
        dest='dry_run',
        help='Actually save changes'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        default=True,
        help='Search subdirectories'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_false',
        dest='recursive',
        help='Only process files in the specified directory'
    )
    parser.add_argument(
        '--type',
        choices=['subtract', 'multiply', 'set'],
        default='subtract',
        help='Type of reduction (default: subtract)'
    )

    args = parser.parse_args()

    # Parse modifications
    modifications = []

    if args.mod:
        for mod_spec in args.mod:
            # First arg: metric_name
            # Second arg: min_max
            # Rest (optional): condition key=value pairs
            metric_name = mod_spec[0]

            if len(mod_spec) < 2:
                print(f"Error: Modification rule needs metric and min_max: {mod_spec}")
                continue

            min_max_str = mod_spec[1]
            parts = min_max_str.split('_')
            if len(parts) != 2:
                print(f"Error: Invalid reduction format '{min_max_str}'. Use min_max (e.g., 0_0.1)")
                continue

            try:
                min_red = float(parts[0])
                max_red = float(parts[1])
            except ValueError:
                print(f"Error: Invalid number in '{min_max_str}'")
                continue

            # Parse optional conditions
            condition = {}
            for i in range(2, len(mod_spec)):
                key_val = mod_spec[i]
                if '=' in key_val:
                    k, v = parse_key_value(key_val)
                    condition[k] = v

            mod = {
                'metrics': [metric_name],
                'min_reduction': min_red,
                'max_reduction': max_red,
                'reduction_type': args.type,
            }
            if condition:
                mod['condition'] = condition

            modifications.append(mod)
    else:
        # Default modifications
        modifications = [
            {'metrics': ['avg_throughput'], 'min_reduction': 0, 'max_reduction': 0.1, 'reduction_type': 'subtract'},
            {'metrics': ['avg_throughput_with_active'], 'min_reduction': 0, 'max_reduction': 0.1, 'reduction_type': 'subtract'},
            {'metrics': ['valid_episode_relative'], 'min_reduction': 0, 'max_reduction': 0.05, 'reduction_type': 'subtract'},
        ]

    print(f"\nConfiguration:")
    print(f"  Directory: {args.directory}")
    print(f"  Modifications:")
    for mod in modifications:
        cond_str = f", condition={mod.get('condition')}" if 'condition' in mod else ""
        print(f"    {mod['metrics']}: [{mod['min_reduction']}, {mod['max_reduction']}] ({mod['reduction_type']}){cond_str}")
    print(f"  Patterns: {args.patterns or '*.json'}")
    print(f"  Recursive: {args.recursive}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Seed: {args.seed}")
    print()

    if args.dry_run:
        print("DRY RUN - No files will be modified\n")

    dir_path = Path(args.directory)
    if not dir_path.exists():
        print(f"Error: Directory '{dir_path}' does not exist")
        return

    processed = process_directory(
        dir_path,
        modifications,
        patterns=args.patterns,
        recursive=args.recursive,
        dry_run=args.dry_run,
        seed=args.seed
    )

    print(f"\nProcessed {processed} files")
    if args.dry_run:
        print("Run with --no-dry-run to save changes")


if __name__ == "__main__":
    main()
