import requests
import pandas as pd
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

# Configuration
API_URL = "http://localhost:8000"
OUTPUT_DIR = Path("benchmark_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / f'benchmark_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# ALGORITHM CONFIGURATIONS
# ============================================================================

CSP_CONFIGS = {
    "mrv_lcv_ac3": {
        "variable_heuristic": "mrv",
        "value_heuristic": "lcv",
        "use_ac3": True
    },
    "mrv_lcv_no_ac3": {
        "variable_heuristic": "mrv",
        "value_heuristic": "lcv",
        "use_ac3": False
    },
    "mrv_most_common_ac3": {
        "variable_heuristic": "mrv",
        "value_heuristic": "most_common",
        "use_ac3": True
    },
    "degree_lcv_ac3": {
        "variable_heuristic": "degree",
        "value_heuristic": "lcv",
        "use_ac3": True
    },
    "mrv_degree_lcv_ac3": {
        "variable_heuristic": "mrv_degree",
        "value_heuristic": "lcv",
        "use_ac3": True
    },
    "none_none_ac3": {
        "variable_heuristic": "none",
        "value_heuristic": "none",
        "use_ac3": True
    },
    "mrv_none_ac3": {
        "variable_heuristic": "mrv",
        "value_heuristic": "none",
        "use_ac3": True
    },
}

GA_CONFIGS = {
    "small_fast": {
        "pop_size": 50,
        "elite_size": 10,
        "mutation_rate": 0.2,
        "crossover_rate": 0.7,
        "tournament_size": 3,
        "generations_per_guess": 10
    },
    "standard": {
        "pop_size": 100,
        "elite_size": 20,
        "mutation_rate": 0.15,
        "crossover_rate": 0.8,
        "tournament_size": 5,
        "generations_per_guess": 15
    },
    "large_thorough": {
        "pop_size": 200,
        "elite_size": 40,
        "mutation_rate": 0.1,
        "crossover_rate": 0.85,
        "tournament_size": 7,
        "generations_per_guess": 20
    },
    "high_mutation": {
        "pop_size": 100,
        "elite_size": 15,
        "mutation_rate": 0.3,
        "crossover_rate": 0.7,
        "tournament_size": 5,
        "generations_per_guess": 15
    },
    "high_elite": {
        "pop_size": 150,
        "elite_size": 50,
        "mutation_rate": 0.15,
        "crossover_rate": 0.8,
        "tournament_size": 5,
        "generations_per_guess": 15
    },
    "quick_evolve": {
        "pop_size": 80,
        "elite_size": 15,
        "mutation_rate": 0.2,
        "crossover_rate": 0.75,
        "tournament_size": 4,
        "generations_per_guess": 5
    },
}

ASTAR_CONFIGS = {
    "narrow_optimal": {
        "beam_width": 50,
        "heuristic_weight": 1.0
    },
    "standard": {
        "beam_width": 100,
        "heuristic_weight": 1.0
    },
    "wide_optimal": {
        "beam_width": 200,
        "heuristic_weight": 1.0
    },
    "weighted_fast": {
        "beam_width": 100,
        "heuristic_weight": 1.5
    },
    "aggressive": {
        "beam_width": 150,
        "heuristic_weight": 2.0
    },
    "very_narrow": {
        "beam_width": 20,
        "heuristic_weight": 1.0
    },
}

SA_CONFIGS = {
    "hot_slow": {
        "initial_temp": 150.0,
        "cooling_rate": 0.98,
        "min_temp": 0.01,
        "iterations_per_temp": 50,
        "reheat_threshold": 0.1
    },
    "standard": {
        "initial_temp": 100.0,
        "cooling_rate": 0.95,
        "min_temp": 0.01,
        "iterations_per_temp": 50,
        "reheat_threshold": 0.1
    },
    "cool_fast": {
        "initial_temp": 80.0,
        "cooling_rate": 0.90,
        "min_temp": 0.01,
        "iterations_per_temp": 50,
        "reheat_threshold": 0.1
    },
    "high_iterations": {
        "initial_temp": 100.0,
        "cooling_rate": 0.95,
        "min_temp": 0.01,
        "iterations_per_temp": 100,
        "reheat_threshold": 0.1
    },
    "aggressive_reheat": {
        "initial_temp": 100.0,
        "cooling_rate": 0.95,
        "min_temp": 0.01,
        "iterations_per_temp": 50,
        "reheat_threshold": 0.3
    },
}

# Attribute combinations to test
ATTRIBUTE_COMBINATIONS = {
    "minimal": ["Type1", "Generation"],
    "basic": ["Type1", "Type2", "Generation"],
    "standard": ["Type1", "Type2", "Generation", "Color"],
    "extended": ["Type1", "Type2", "Generation", "Color", "Height"],
    "comprehensive": ["Type1", "Type2", "Generation", "Color", "Height", "evolutionary_stage"],
    "numeric": ["Generation", "Height", "Weight"],
    "types_only": ["Type1", "Type2"],
}


# ============================================================================
# API FUNCTIONS
# ============================================================================

def get_pokemon_list() -> List[Dict[str, str]]:
    """Fetch list of all Pokemon from API."""
    try:
        response = requests.get(f"{API_URL}/pokemon", timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("pokemon", [])
    except Exception as e:
        logger.error(f"Failed to fetch Pokemon list: {e}")
        return []


def run_solver(algorithm: str, attributes: List[str], secret_pokemon: str,
               config: Dict[str, Any], max_attempts: int = 10) -> Optional[Dict[str, Any]]:
    """Run solver with specified configuration."""
    payload = {
        "algorithm": algorithm,
        "attributes": attributes,
        "secret_pokemon": secret_pokemon,
        "max_attempts": max_attempts
    }
    
    # Add algorithm-specific config
    if algorithm == "CSP":
        payload["csp_config"] = config
    elif algorithm == "GA":
        payload["ga_config"] = config
    elif algorithm == "ASTAR":
        payload["astar_config"] = config
    elif algorithm == "SA":
        payload["sa_config"] = config
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/solve",
            json=payload,
            timeout=300  # 5 minute timeout
        )
        wall_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            result['wall_time'] = wall_time  # Add actual wall time
            return result
        else:
            logger.error(f"Solver failed with status {response.status_code}: {response.text}")
            return None
    except requests.exceptions.Timeout:
        logger.error(f"Solver timeout for {algorithm} on {secret_pokemon}")
        return None
    except Exception as e:
        logger.error(f"Solver error: {e}")
        return None


# ============================================================================
# BENCHMARKING FUNCTIONS
# ============================================================================

def benchmark_single_run(
    algorithm: str,
    config_name: str,
    config: Dict[str, Any],
    attributes: List[str],
    attribute_name: str,
    pokemon: str,
    run_number: int,
    max_attempts: int = 10
) -> Dict[str, Any]:
    """Run a single benchmark test."""
    logger.info(f"Run {run_number}: {algorithm}/{config_name}/{attribute_name} on {pokemon}")
    
    result = run_solver(algorithm, attributes, pokemon, config, max_attempts)
    
    if result:
        return {
            'timestamp': datetime.now().isoformat(),
            'run_number': run_number,
            'algorithm': algorithm,
            'config_name': config_name,
            'attribute_set': attribute_name,
            'attributes': ','.join(attributes),
            'num_attributes': len(attributes),
            'pokemon': pokemon,
            'success': result.get('success', False),
            'total_attempts': result.get('total_attempts', 0),
            'execution_time': result.get('execution_time', 0),
            'wall_time': result.get('wall_time', 0),
            'efficiency': result.get('performance_metrics', {}).get('efficiency', 0),
            'convergence_rate': result.get('performance_metrics', {}).get('convergence_rate', 0),
            'avg_time_per_guess': result.get('performance_metrics', {}).get('avg_time_per_guess', 0),
            **config  # Include all config parameters
        }
    else:
        return {
            'timestamp': datetime.now().isoformat(),
            'run_number': run_number,
            'algorithm': algorithm,
            'config_name': config_name,
            'attribute_set': attribute_name,
            'attributes': ','.join(attributes),
            'num_attributes': len(attributes),
            'pokemon': pokemon,
            'success': False,
            'total_attempts': 0,
            'execution_time': 0,
            'wall_time': 0,
            'efficiency': 0,
            'convergence_rate': 0,
            'avg_time_per_guess': 0,
            'error': 'solver_failed',
            **config
        }


def benchmark_algorithm(
    algorithm: str,
    configs: Dict[str, Dict[str, Any]],
    attribute_combinations: Dict[str, List[str]],
    pokemon_list: List[str],
    num_runs: int = 1,
    max_attempts: int = 10,
    parallel: bool = False
) -> List[Dict[str, Any]]:
    """Benchmark an algorithm with all specified configurations."""
    results = []
    total_tests = len(configs) * len(attribute_combinations) * len(pokemon_list) * num_runs
    
    logger.info(f"Starting {algorithm} benchmark: {total_tests} total tests")
    
    # Generate all test cases
    test_cases = []
    for config_name, config in configs.items():
        for attr_name, attributes in attribute_combinations.items():
            for pokemon in pokemon_list:
                for run_num in range(1, num_runs + 1):
                    test_cases.append((
                        algorithm, config_name, config, attributes,
                        attr_name, pokemon, run_num, max_attempts
                    ))
    
    # Run tests
    if parallel:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(benchmark_single_run, *test_case)
                for test_case in test_cases
            ]
            
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Progress: {i}/{total_tests} ({i/total_tests*100:.1f}%)")
                except Exception as e:
                    logger.error(f"Test failed: {e}")
    else:
        for i, test_case in enumerate(test_cases, 1):
            try:
                result = benchmark_single_run(*test_case)
                results.append(result)
                logger.info(f"Progress: {i}/{total_tests} ({i/total_tests*100:.1f}%)")
            except Exception as e:
                logger.error(f"Test failed: {e}")
    
    return results


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_results(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive analysis of benchmark results."""
    analysis = {
        'summary': {},
        'by_algorithm': {},
        'by_config': {},
        'by_attributes': {},
        'best_configs': {}
    }
    
    # Overall summary
    analysis['summary'] = {
        'total_runs': len(df),
        'successful_runs': df['success'].sum(),
        'success_rate': df['success'].mean(),
        'avg_attempts': df[df['success']]['total_attempts'].mean(),
        'avg_time': df[df['success']]['execution_time'].mean(),
        'avg_efficiency': df[df['success']]['efficiency'].mean(),
    }
    
    # By algorithm
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        analysis['by_algorithm'][algo] = {
            'runs': len(algo_df),
            'success_rate': algo_df['success'].mean(),
            'avg_attempts': algo_df[algo_df['success']]['total_attempts'].mean(),
            'avg_time': algo_df[algo_df['success']]['execution_time'].mean(),
            'std_attempts': algo_df[algo_df['success']]['total_attempts'].std(),
            'min_attempts': algo_df[algo_df['success']]['total_attempts'].min(),
            'max_attempts': algo_df[algo_df['success']]['total_attempts'].max(),
        }
    
    # By config
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        analysis['by_config'][algo] = {}
        for config in algo_df['config_name'].unique():
            config_df = algo_df[algo_df['config_name'] == config]
            analysis['by_config'][algo][config] = {
                'success_rate': config_df['success'].mean(),
                'avg_attempts': config_df[config_df['success']]['total_attempts'].mean(),
                'avg_time': config_df[config_df['success']]['execution_time'].mean(),
            }
    
    # By attribute set
    for attr_set in df['attribute_set'].unique():
        attr_df = df[df['attribute_set'] == attr_set]
        analysis['by_attributes'][attr_set] = {
            'num_attributes': attr_df['num_attributes'].iloc[0],
            'success_rate': attr_df['success'].mean(),
            'avg_attempts': attr_df[attr_df['success']]['total_attempts'].mean(),
            'avg_time': attr_df[attr_df['success']]['execution_time'].mean(),
        }
    
    # Best configs per algorithm
    for algo in df['algorithm'].unique():
        algo_df = df[(df['algorithm'] == algo) & (df['success'] == True)]
        if len(algo_df) > 0:
            best_config = algo_df.groupby('config_name').agg({
                'total_attempts': 'mean',
                'execution_time': 'mean',
                'success': 'mean'
            }).sort_values('total_attempts').iloc[0]
            
            analysis['best_configs'][algo] = {
                'config': best_config.name,
                'avg_attempts': best_config['total_attempts'],
                'avg_time': best_config['execution_time'],
                'success_rate': best_config['success']
            }
    
    return analysis


def generate_report(df: pd.DataFrame, analysis: Dict[str, Any], output_path: Path):
    """Generate comprehensive report."""
    report_path = output_path.with_suffix('.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("POKEDLE AI SOLVER - BENCHMARK REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Total Runs: {analysis['summary']['total_runs']}\n\n")
        
        # Overall Summary
        f.write("OVERALL SUMMARY\n")
        f.write("-" * 80 + "\n")
        for key, value in analysis['summary'].items():
            f.write(f"{key:30s}: {value:.3f}\n")
        f.write("\n")
        
        # Algorithm Comparison
        f.write("ALGORITHM COMPARISON\n")
        f.write("-" * 80 + "\n")
        for algo, stats in analysis['by_algorithm'].items():
            f.write(f"\n{algo}:\n")
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {key:25s}: {value:.3f}\n")
                else:
                    f.write(f"  {key:25s}: {value}\n")
        f.write("\n")
        
        # Best Configurations
        f.write("BEST CONFIGURATIONS PER ALGORITHM\n")
        f.write("-" * 80 + "\n")
        for algo, best in analysis['best_configs'].items():
            f.write(f"\n{algo}: {best['config']}\n")
            f.write(f"  Avg Attempts: {best['avg_attempts']:.2f}\n")
            f.write(f"  Avg Time:     {best['avg_time']:.3f}s\n")
            f.write(f"  Success Rate: {best['success_rate']*100:.1f}%\n")
        f.write("\n")
        
        # Attribute Set Analysis
        f.write("ATTRIBUTE SET ANALYSIS\n")
        f.write("-" * 80 + "\n")
        for attr_set, stats in analysis['by_attributes'].items():
            f.write(f"\n{attr_set} ({stats['num_attributes']} attributes):\n")
            f.write(f"  Success Rate: {stats['success_rate']*100:.1f}%\n")
            f.write(f"  Avg Attempts: {stats['avg_attempts']:.2f}\n")
            f.write(f"  Avg Time:     {stats['avg_time']:.3f}s\n")
    
    logger.info(f"Report saved to {report_path}")


# ============================================================================
# MAIN BENCHMARK MODES
# ============================================================================

def quick_benchmark(pokemon_sample_size: int = 10, num_runs: int = 1):
    """Quick benchmark with minimal configurations."""
    logger.info("Starting QUICK benchmark mode")
    
    pokemon_list = get_pokemon_list()
    if not pokemon_list:
        logger.error("Failed to fetch Pokemon list")
        return
    
    # Sample random Pokemon
    import random
    pokemon_sample = random.sample(
        [p['name'] for p in pokemon_list],
        min(pokemon_sample_size, len(pokemon_list))
    )
    
    # Minimal configs
    configs_to_test = {
        'CSP': {'standard': CSP_CONFIGS['mrv_lcv_ac3']},
        'GA': {'standard': GA_CONFIGS['small_fast']},
        'ASTAR': {'standard': ASTAR_CONFIGS['standard']},
        'SA': {'standard': SA_CONFIGS['standard']}
    }
    
    # Minimal attribute sets
    attr_sets = {
        'basic': ATTRIBUTE_COMBINATIONS['basic'],
        'standard': ATTRIBUTE_COMBINATIONS['standard']
    }
    
    all_results = []
    for algo, configs in configs_to_test.items():
        results = benchmark_algorithm(
            algo, configs, attr_sets, pokemon_sample,
            num_runs=num_runs, parallel=False
        )
        all_results.extend(results)
    
    # Save results
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = OUTPUT_DIR / f'quick_benchmark_{timestamp}.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    # Generate analysis
    analysis = analyze_results(df)
    generate_report(df, analysis, output_path)
    
    return df, analysis


def standard_benchmark(pokemon_sample_size: int = 50, num_runs: int = 3):
    """Standard benchmark with common configurations."""
    logger.info("Starting STANDARD benchmark mode")
    
    pokemon_list = get_pokemon_list()
    if not pokemon_list:
        logger.error("Failed to fetch Pokemon list")
        return
    
    import random
    pokemon_sample = random.sample(
        [p['name'] for p in pokemon_list],
        min(pokemon_sample_size, len(pokemon_list))
    )
    
    # Standard configs
    configs_to_test = {
        'CSP': {
            'mrv_lcv_ac3': CSP_CONFIGS['mrv_lcv_ac3'],
            'degree_lcv_ac3': CSP_CONFIGS['degree_lcv_ac3'],
            'mrv_degree_lcv_ac3': CSP_CONFIGS['mrv_degree_lcv_ac3'],
        },
        'GA': {
            'small_fast': GA_CONFIGS['small_fast'],
            'standard': GA_CONFIGS['standard'],
            'high_mutation': GA_CONFIGS['high_mutation'],
        },
        'ASTAR': {
            'narrow_optimal': ASTAR_CONFIGS['narrow_optimal'],
            'standard': ASTAR_CONFIGS['standard'],
            'wide_optimal': ASTAR_CONFIGS['wide_optimal'],
        },
        'SA': {
            'standard': SA_CONFIGS['standard'],
            'cool_fast': SA_CONFIGS['cool_fast'],
            'high_iterations': SA_CONFIGS['high_iterations'],
        }
    }
    
    attr_sets = {
        'minimal': ATTRIBUTE_COMBINATIONS['minimal'],
        'basic': ATTRIBUTE_COMBINATIONS['basic'],
        'standard': ATTRIBUTE_COMBINATIONS['standard'],
        'extended': ATTRIBUTE_COMBINATIONS['extended'],
    }
    
    all_results = []
    for algo, configs in configs_to_test.items():
        results = benchmark_algorithm(
            algo, configs, attr_sets, pokemon_sample,
            num_runs=num_runs, parallel=False
        )
        all_results.extend(results)
    
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = OUTPUT_DIR / f'standard_benchmark_{timestamp}.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    analysis = analyze_results(df)
    generate_report(df, analysis, output_path)
    
    return df, analysis


def comprehensive_benchmark(pokemon_sample_size: int = 100, num_runs: int = 5):
    """Comprehensive benchmark with all configurations."""
    logger.info("Starting COMPREHENSIVE benchmark mode")
    
    pokemon_list = get_pokemon_list()
    if not pokemon_list:
        logger.error("Failed to fetch Pokemon list")
        return
    
    import random
    pokemon_sample = random.sample(
        [p['name'] for p in pokemon_list],
        min(pokemon_sample_size, len(pokemon_list))
    )
    
    # All configs
    configs_to_test = {
        'CSP': CSP_CONFIGS,
        'GA': GA_CONFIGS,
        'ASTAR': ASTAR_CONFIGS,
        'SA': SA_CONFIGS
    }
    
    attr_sets = ATTRIBUTE_COMBINATIONS
    
    all_results = []
    for algo, configs in configs_to_test.items():
        results = benchmark_algorithm(
            algo, configs, attr_sets, pokemon_sample,
            num_runs=num_runs, parallel=False
        )
        all_results.extend(results)
    
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = OUTPUT_DIR / f'comprehensive_benchmark_{timestamp}.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    analysis = analyze_results(df)
    generate_report(df, analysis, output_path)
    
    return df, analysis


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Pokedle AI Solver algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --mode quick
  python benchmark.py --mode standard --pokemon 50 --runs 3
  python benchmark.py --mode comprehensive --pokemon 100
  python benchmark.py --algorithms CSP GA --pokemon 20
        """
    )
    
    parser.add_argument('--mode', choices=['quick', 'standard', 'comprehensive', 'custom'],
                       default='quick', help='Benchmark mode')
    parser.add_argument('--algorithms', nargs='+', choices=['CSP', 'GA', 'ASTAR', 'SA'],
                       help='Algorithms to test (custom mode)')
    parser.add_argument('--pokemon', type=int, default=10,
                       help='Number of Pokemon to test')
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of runs per configuration')
    parser.add_argument('--output', type=str,
                       help='Output file path')
    parser.add_argument('--parallel', action='store_true',
                       help='Run tests in parallel (experimental)')
    
    args = parser.parse_args()
    
    # Check API connectivity
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        logger.info("API connection successful")
    except Exception as e:
        logger.error(f"Cannot connect to API at {API_URL}: {e}")
        logger.error("Make sure backend is running: cd backend && python main.py")
        return
    
    # Run benchmark
    if args.mode == 'quick':
        df, analysis = quick_benchmark(args.pokemon, args.runs)
    elif args.mode == 'standard':
        df, analysis = standard_benchmark(args.pokemon, args.runs)
    elif args.mode == 'comprehensive':
        df, analysis = comprehensive_benchmark(args.pokemon, args.runs)
    elif args.mode == 'custom':
        if not args.algorithms:
            logger.error("--algorithms required for custom mode")
            return
        # Implement custom mode here
        logger.info("Custom mode not yet implemented")
    
    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()