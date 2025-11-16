import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import numpy as np
from typing import List

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_results(filepath: str) -> pd.DataFrame:
    """Load benchmark results from CSV."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} results from {filepath}")
    return df


def plot_algorithm_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot comparison across algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Success Rate
    success_by_algo = df.groupby('algorithm')['success'].mean()
    axes[0, 0].bar(success_by_algo.index, success_by_algo.values)
    axes[0, 0].set_title('Success Rate by Algorithm')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_ylim([0, 1])
    for i, v in enumerate(success_by_algo.values):
        axes[0, 0].text(i, v + 0.02, f'{v:.2%}', ha='center')
    
    # Average Attempts (successful runs only)
    attempts_by_algo = df[df['success']].groupby('algorithm')['total_attempts'].mean()
    axes[0, 1].bar(attempts_by_algo.index, attempts_by_algo.values, color='orange')
    axes[0, 1].set_title('Average Attempts (Successful Runs)')
    axes[0, 1].set_ylabel('Attempts')
    for i, v in enumerate(attempts_by_algo.values):
        axes[0, 1].text(i, v + 0.1, f'{v:.2f}', ha='center')
    
    # Execution Time
    time_by_algo = df[df['success']].groupby('algorithm')['execution_time'].mean()
    axes[1, 0].bar(time_by_algo.index, time_by_algo.values, color='green')
    axes[1, 0].set_title('Average Execution Time (Successful Runs)')
    axes[1, 0].set_ylabel('Time (seconds)')
    for i, v in enumerate(time_by_algo.values):
        axes[1, 0].text(i, v + 0.05, f'{v:.2f}s', ha='center')
    
    # Efficiency
    efficiency_by_algo = df[df['success']].groupby('algorithm')['efficiency'].mean()
    axes[1, 1].bar(efficiency_by_algo.index, efficiency_by_algo.values, color='purple')
    axes[1, 1].set_title('Average Efficiency (Successful Runs)')
    axes[1, 1].set_ylabel('Efficiency Score')
    for i, v in enumerate(efficiency_by_algo.values):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'algorithm_comparison.png'}")
    plt.close()


def plot_config_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot configuration comparison within each algorithm."""
    algorithms = df['algorithm'].unique()
    
    for algo in algorithms:
        algo_df = df[df['algorithm'] == algo]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'{algo} Configuration Comparison', fontsize=16)
        
        # Success Rate by Config
        success_by_config = algo_df.groupby('config_name')['success'].mean().sort_values(ascending=False)
        axes[0].barh(success_by_config.index, success_by_config.values)
        axes[0].set_xlabel('Success Rate')
        axes[0].set_title('Success Rate by Configuration')
        axes[0].set_xlim([0, 1])
        
        # Average Attempts by Config
        attempts_by_config = algo_df[algo_df['success']].groupby('config_name')['total_attempts'].mean().sort_values()
        axes[1].barh(attempts_by_config.index, attempts_by_config.values, color='orange')
        axes[1].set_xlabel('Average Attempts')
        axes[1].set_title('Attempts by Configuration (Successful Runs)')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{algo}_config_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / f'{algo}_config_comparison.png'}")
        plt.close()


def plot_attribute_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze performance by attribute set."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Success rate by number of attributes
    success_by_attrs = df.groupby('num_attributes')['success'].mean()
    axes[0, 0].plot(success_by_attrs.index, success_by_attrs.values, marker='o', linewidth=2)
    axes[0, 0].set_xlabel('Number of Attributes')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_title('Success Rate vs Number of Attributes')
    axes[0, 0].grid(True)
    
    # Attempts by number of attributes
    attempts_by_attrs = df[df['success']].groupby('num_attributes')['total_attempts'].mean()
    axes[0, 1].plot(attempts_by_attrs.index, attempts_by_attrs.values, marker='o', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Number of Attributes')
    axes[0, 1].set_ylabel('Average Attempts')
    axes[0, 1].set_title('Attempts vs Number of Attributes')
    axes[0, 1].grid(True)
    
    # Time by number of attributes
    time_by_attrs = df[df['success']].groupby('num_attributes')['execution_time'].mean()
    axes[1, 0].plot(time_by_attrs.index, time_by_attrs.values, marker='o', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Number of Attributes')
    axes[1, 0].set_ylabel('Execution Time (s)')
    axes[1, 0].set_title('Execution Time vs Number of Attributes')
    axes[1, 0].grid(True)
    
    # Heatmap: Algorithm vs Attribute Set
    pivot = df[df['success']].pivot_table(
        values='total_attempts',
        index='algorithm',
        columns='attribute_set',
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd_r', ax=axes[1, 1])
    axes[1, 1].set_title('Average Attempts Heatmap')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attribute_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'attribute_analysis.png'}")
    plt.close()


def plot_convergence_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze convergence characteristics."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Convergence rate by algorithm
    convergence_by_algo = df[df['success']].groupby('algorithm')['convergence_rate'].mean()
    axes[0].bar(convergence_by_algo.index, convergence_by_algo.values, color='teal')
    axes[0].set_ylabel('Convergence Rate')
    axes[0].set_title('Average Convergence Rate by Algorithm')
    axes[0].set_ylim([0, convergence_by_algo.max() * 1.2])
    
    # Scatter: Attempts vs Time
    for algo in df['algorithm'].unique():
        algo_df = df[(df['algorithm'] == algo) & (df['success'])]
        axes[1].scatter(
            algo_df['total_attempts'],
            algo_df['execution_time'],
            label=algo,
            alpha=0.6,
            s=50
        )
    axes[1].set_xlabel('Total Attempts')
    axes[1].set_ylabel('Execution Time (s)')
    axes[1].set_title('Attempts vs Time (Successful Runs)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'convergence_analysis.png'}")
    plt.close()


def plot_distribution_analysis(df: pd.DataFrame, output_dir: Path):
    """Plot distribution of attempts and times."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Attempts distribution
    for algo in df['algorithm'].unique():
        algo_df = df[(df['algorithm'] == algo) & (df['success'])]
        axes[0, 0].hist(
            algo_df['total_attempts'],
            alpha=0.5,
            label=algo,
            bins=range(1, 15)
        )
    axes[0, 0].set_xlabel('Total Attempts')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Attempts')
    axes[0, 0].legend()
    
    # Time distribution
    for algo in df['algorithm'].unique():
        algo_df = df[(df['algorithm'] == algo) & (df['success'])]
        axes[0, 1].hist(
            algo_df['execution_time'],
            alpha=0.5,
            label=algo,
            bins=20
        )
    axes[0, 1].set_xlabel('Execution Time (s)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Execution Times')
    axes[0, 1].legend()
    
    # Box plot: Attempts
    df[df['success']].boxplot(
        column='total_attempts',
        by='algorithm',
        ax=axes[1, 0]
    )
    axes[1, 0].set_xlabel('Algorithm')
    axes[1, 0].set_ylabel('Total Attempts')
    axes[1, 0].set_title('Attempts Distribution by Algorithm')
    plt.sca(axes[1, 0])
    plt.xticks(rotation=0)
    
    # Box plot: Time
    df[df['success']].boxplot(
        column='execution_time',
        by='algorithm',
        ax=axes[1, 1]
    )
    axes[1, 1].set_xlabel('Algorithm')
    axes[1, 1].set_ylabel('Execution Time (s)')
    axes[1, 1].set_title('Time Distribution by Algorithm')
    plt.sca(axes[1, 1])
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'distribution_analysis.png'}")
    plt.close()


def generate_summary_stats(df: pd.DataFrame, output_path: Path):
    """Generate comprehensive summary statistics."""
    summary = []
    
    summary.append("=" * 80)
    summary.append("BENCHMARK SUMMARY STATISTICS")
    summary.append("=" * 80)
    summary.append(f"\nTotal Runs: {len(df)}")
    summary.append(f"Successful Runs: {df['success'].sum()} ({df['success'].mean():.1%})")
    summary.append(f"Failed Runs: {(~df['success']).sum()} ({(~df['success']).mean():.1%})")
    
    # Overall stats
    summary.append("\n" + "-" * 80)
    summary.append("OVERALL STATISTICS (Successful Runs)")
    summary.append("-" * 80)
    successful_df = df[df['success']]
    summary.append(f"Average Attempts:     {successful_df['total_attempts'].mean():.2f} ± {successful_df['total_attempts'].std():.2f}")
    summary.append(f"Median Attempts:      {successful_df['total_attempts'].median():.0f}")
    summary.append(f"Min Attempts:         {successful_df['total_attempts'].min():.0f}")
    summary.append(f"Max Attempts:         {successful_df['total_attempts'].max():.0f}")
    summary.append(f"Average Time:         {successful_df['execution_time'].mean():.3f}s ± {successful_df['execution_time'].std():.3f}s")
    summary.append(f"Average Efficiency:   {successful_df['efficiency'].mean():.3f}")
    
    # Per algorithm
    summary.append("\n" + "-" * 80)
    summary.append("PER ALGORITHM STATISTICS")
    summary.append("-" * 80)
    
    for algo in sorted(df['algorithm'].unique()):
        algo_df = df[df['algorithm'] == algo]
        algo_success_df = algo_df[algo_df['success']]
        
        summary.append(f"\n{algo}:")
        summary.append(f"  Total Runs:         {len(algo_df)}")
        summary.append(f"  Success Rate:       {algo_df['success'].mean():.1%}")
        if len(algo_success_df) > 0:
            summary.append(f"  Avg Attempts:       {algo_success_df['total_attempts'].mean():.2f} ± {algo_success_df['total_attempts'].std():.2f}")
            summary.append(f"  Median Attempts:    {algo_success_df['total_attempts'].median():.0f}")
            summary.append(f"  Avg Time:           {algo_success_df['execution_time'].mean():.3f}s")
            summary.append(f"  Avg Efficiency:     {algo_success_df['efficiency'].mean():.3f}")
        
        # Best config for this algorithm
        if len(algo_success_df) > 0:
            best_config = algo_success_df.groupby('config_name').agg({
                'total_attempts': 'mean',
                'success': 'mean'
            }).sort_values('total_attempts').iloc[0]
            summary.append(f"  Best Config:        {best_config.name} ({best_config['total_attempts']:.2f} avg attempts)")
    
    # Attribute analysis
    summary.append("\n" + "-" * 80)
    summary.append("ATTRIBUTE SET ANALYSIS")
    summary.append("-" * 80)
    
    for attr_set in sorted(df['attribute_set'].unique()):
        attr_df = df[df['attribute_set'] == attr_set]
        attr_success_df = attr_df[attr_df['success']]
        
        summary.append(f"\n{attr_set} ({attr_df['num_attributes'].iloc[0]} attributes):")
        summary.append(f"  Success Rate:       {attr_df['success'].mean():.1%}")
        if len(attr_success_df) > 0:
            summary.append(f"  Avg Attempts:       {attr_success_df['total_attempts'].mean():.2f}")
            summary.append(f"  Avg Time:           {attr_success_df['execution_time'].mean():.3f}s")
    
    # Save to file
    summary_text = "\n".join(summary)
    summary_path = output_path / 'summary_statistics.txt'
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\nSummary saved to: {summary_path}")


def compare_multiple_benchmarks(filepaths: List[str], output_dir: Path):
    """Compare results from multiple benchmark runs."""
    dfs = []
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        df['benchmark_file'] = Path(filepath).stem
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparison Across Multiple Benchmark Runs', fontsize=16)
    
    # Success rate comparison
    pivot_success = combined_df.groupby(['benchmark_file', 'algorithm'])['success'].mean().unstack()
    pivot_success.plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Success Rate Comparison')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].legend(title='Algorithm')
    axes[0, 0].set_ylim([0, 1])
    
    # Attempts comparison
    pivot_attempts = combined_df[combined_df['success']].groupby(['benchmark_file', 'algorithm'])['total_attempts'].mean().unstack()
    pivot_attempts.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Average Attempts Comparison')
    axes[0, 1].set_ylabel('Attempts')
    axes[0, 1].legend(title='Algorithm')
    
    # Time comparison
    pivot_time = combined_df[combined_df['success']].groupby(['benchmark_file', 'algorithm'])['execution_time'].mean().unstack()
    pivot_time.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Average Time Comparison')
    axes[1, 0].set_ylabel('Time (s)')
    axes[1, 0].legend(title='Algorithm')
    
    # Efficiency comparison
    pivot_eff = combined_df[combined_df['success']].groupby(['benchmark_file', 'algorithm'])['efficiency'].mean().unstack()
    pivot_eff.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Average Efficiency Comparison')
    axes[1, 1].set_ylabel('Efficiency')
    axes[1, 1].legend(title='Algorithm')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'multi_benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'multi_benchmark_comparison.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('files', nargs='+', help='Benchmark CSV files to analyze')
    parser.add_argument('--compare', action='store_true', help='Compare multiple files')
    parser.add_argument('--output', type=str, help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.files[0]).parent / 'analysis'
    
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    if args.compare and len(args.files) > 1:
        # Compare multiple benchmarks
        print(f"\nComparing {len(args.files)} benchmark files...")
        compare_multiple_benchmarks(args.files, output_dir)
    else:
        # Analyze single benchmark
        print(f"\nAnalyzing: {args.files[0]}")
        df = load_results(args.files[0])
        
        print("\nGenerating visualizations...")
        plot_algorithm_comparison(df, output_dir)
        plot_config_comparison(df, output_dir)
        plot_attribute_analysis(df, output_dir)
        plot_convergence_analysis(df, output_dir)
        plot_distribution_analysis(df, output_dir)
        
        print("\nGenerating summary statistics...")
        generate_summary_stats(df, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()