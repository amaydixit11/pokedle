from typing import Dict, List, Any
from dataclasses import dataclass
import statistics

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    avg_time_per_guess: float
    total_guesses: int
    success_rate: float
    efficiency: float
    convergence_rate: float = 0.0
    diversity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "avg_time_per_guess": round(self.avg_time_per_guess, 3),
            "total_guesses": self.total_guesses,
            "success_rate": round(self.success_rate, 3),
            "efficiency": round(self.efficiency, 3),
            "convergence_rate": round(self.convergence_rate, 3),
            "diversity_score": round(self.diversity_score, 3)
        }

def calculate_metrics(steps: List[Any], execution_time: float, success: bool) -> PerformanceMetrics:
    """
    Calculate performance metrics from solver steps.
    
    Args:
        steps: List of solver steps
        execution_time: Total execution time
        success: Whether the solver succeeded
        
    Returns:
        PerformanceMetrics object
    """
    if not steps:
        return PerformanceMetrics(
            avg_time_per_guess=0,
            total_guesses=0,
            success_rate=0.0,
            efficiency=0.0
        )
    
    # Basic metrics
    total_guesses = len(steps)
    avg_time = execution_time / total_guesses
    success_rate = 1.0 if success else 0.0
    efficiency = 1.0 / total_guesses if total_guesses > 0 else 0.0
    
    # Convergence rate (how quickly candidates decrease)
    convergence_rate = 0.0
    if len(steps) > 1:
        first_candidates = steps[0].remaining_candidates
        last_candidates = steps[-1].remaining_candidates
        
        if first_candidates > 0:
            reduction_rate = (first_candidates - last_candidates) / first_candidates
            convergence_rate = reduction_rate / len(steps)
    
    # Diversity score (variation in guesses)
    diversity_score = 0.0
    if len(steps) > 1:
        unique_guesses = len(set(step.guess_name for step in steps))
        diversity_score = unique_guesses / len(steps)
    
    return PerformanceMetrics(
        avg_time_per_guess=avg_time,
        total_guesses=total_guesses,
        success_rate=success_rate,
        efficiency=efficiency,
        convergence_rate=convergence_rate,
        diversity_score=diversity_score
    )

def calculate_algorithm_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate aggregate statistics across multiple runs.
    
    Args:
        results: List of result dictionaries from multiple runs
        
    Returns:
        Dictionary with aggregate statistics
    """
    if not results:
        return {}
    
    attempts = [r['total_attempts'] for r in results]
    times = [r['execution_time'] for r in results]
    successes = [r['success'] for r in results]
    
    return {
        "runs": len(results),
        "avg_attempts": round(statistics.mean(attempts), 2),
        "median_attempts": statistics.median(attempts),
        "min_attempts": min(attempts),
        "max_attempts": max(attempts),
        "std_attempts": round(statistics.stdev(attempts), 2) if len(attempts) > 1 else 0,
        "avg_time": round(statistics.mean(times), 3),
        "success_rate": sum(successes) / len(successes),
        "total_time": round(sum(times), 3)
    }

def compare_algorithms(results_by_algorithm: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Compare performance across different algorithms.
    
    Args:
        results_by_algorithm: Dictionary mapping algorithm names to result lists
        
    Returns:
        Comparison dictionary with rankings and statistics
    """
    comparison = {}
    
    for algo_name, results in results_by_algorithm.items():
        stats = calculate_algorithm_statistics(results)
        comparison[algo_name] = stats
    
    # Rank algorithms
    if comparison:
        # Rank by average attempts (lower is better)
        attempts_ranking = sorted(
            comparison.items(),
            key=lambda x: x[1].get('avg_attempts', float('inf'))
        )
        
        # Rank by success rate (higher is better)
        success_ranking = sorted(
            comparison.items(),
            key=lambda x: x[1].get('success_rate', 0),
            reverse=True
        )
        
        return {
            "algorithms": comparison,
            "best_by_attempts": attempts_ranking[0][0] if attempts_ranking else None,
            "best_by_success": success_ranking[0][0] if success_ranking else None,
            "attempts_ranking": [name for name, _ in attempts_ranking],
            "success_ranking": [name for name, _ in success_ranking]
        }
    
    return {"algorithms": comparison}

def calculate_heuristic_efficiency(steps: List[Any]) -> Dict[str, float]:
    """
    Calculate efficiency metrics for heuristic performance.
    
    Args:
        steps: List of solver steps
        
    Returns:
        Dictionary with heuristic efficiency metrics
    """
    if not steps:
        return {}
    
    # Calculate candidate reduction per step
    reductions = []
    for i in range(len(steps) - 1):
        prev_candidates = steps[i].remaining_candidates
        curr_candidates = steps[i + 1].remaining_candidates
        
        if prev_candidates > 0:
            reduction_ratio = (prev_candidates - curr_candidates) / prev_candidates
            reductions.append(reduction_ratio)
    
    avg_reduction = statistics.mean(reductions) if reductions else 0.0
    
    # Calculate information gain per guess
    information_gains = []
    for step in steps:
        if step.remaining_candidates > 0:
            # Information gain = log2(prev_candidates / curr_candidates)
            import math
            gain = math.log2(steps[0].remaining_candidates / step.remaining_candidates) if step.remaining_candidates > 0 else 0
            information_gains.append(gain)
    
    avg_info_gain = statistics.mean(information_gains) if information_gains else 0.0
    
    return {
        "avg_candidate_reduction": round(avg_reduction, 3),
        "avg_information_gain": round(avg_info_gain, 3),
        "total_information_gain": round(sum(information_gains), 3),
        "reduction_consistency": round(statistics.stdev(reductions), 3) if len(reductions) > 1 else 0.0
    }