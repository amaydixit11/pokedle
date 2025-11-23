from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import pandas as pd
from typing import List, Optional, Callable

# Import configurations and models
from config import *
from models import *
from data_loader import DataLoader
from feedback import get_feedback, is_complete_match

# Import corrected algorithms
from algorithms.csp_solver import CSPSolver
from algorithms.ga_solver import GASolver
from algorithms.astar_solver import AStarSolver
from algorithms.simulated_annealing import SimulatedAnnealingSolver

# Import utilities
from utils.metrics import calculate_metrics
from utils.validators import validate_config

import asyncio
import json

import logging

from rich.console import Console
from rich.logging import RichHandler
import logging

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[RichHandler(rich_tracebacks=True)]
)



app = FastAPI(
    title="Pokedle Solver API - Logically Correct Version",
    version="5.0",
    description="AI-powered Pokedle solver with properly implemented CSP, GA, A*, and SA algorithms"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data loader
data_loader = DataLoader()
data_loader.load_data(CSV_PATH)

# ============ Helper Functions ============

def create_solver(config: SolverConfig):
    """
    Factory function to create appropriate solver with corrected implementations.
    """
    df = data_loader.get_dataframe()
    
    if config.algorithm == 'CSP':
        csp_config = config.csp_config or CSPConfig()
        return CSPSolver(
            df, 
            config.attributes,
            variable_heuristic=csp_config.variable_heuristic,
            value_heuristic=csp_config.value_heuristic
        )
    
    elif config.algorithm == 'GA':
        ga_config = config.ga_config or GAConfig()
        return GASolver(df, config.attributes, ga_config.dict())
    
    elif config.algorithm == 'ASTAR':
        astar_config = config.astar_config or AStarConfig()
        return AStarSolver(df, config.attributes, astar_config.dict())
    
    elif config.algorithm == 'SA':
        sa_config = config.sa_config or SAConfig()
        return SimulatedAnnealingSolver(df, config.attributes, sa_config.dict())
    
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

# ============ API Endpoints ============

@app.get("/")
def root():
    return {
        "message": "Pokedle Solver API - Logically Correct Version",
        "version": "5.0",
        "improvements": [
            "CSP: Proper variable/domain/constraint formulation with AC-3",
            "CSP: Separate variable ordering and value ordering heuristics",
            "GA: Valid Pokemon individuals (no arbitrary combinations)",
            "GA: Constraint-based fitness function",
            "A*: Admissible heuristic guaranteeing optimality",
            "All: Theoretically sound implementations"
        ]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "pokemon_loaded": data_loader.pokemon_count,
        "timestamp": time.time(),
        "version": "5.0-corrected"
    }

@app.get("/pokemon")
def get_pokemon_list():
    """Get list of all Pokemon"""
    df = data_loader.get_dataframe()
    pokemon_list = []
    
    for _, row in df.iterrows():
        pokemon_list.append({
            "name": row['Original_Name'],
            "image_url": row.get('image_url', ''),
            "generation": int(row.get('Generation', 0)) if not pd.isna(row.get('Generation')) else None,
            "type1": row.get('Type1'),
            "type2": row.get('Type2') if not pd.isna(row.get('Type2')) else None
        })
    
    return {
        "pokemon": pokemon_list,
        "count": len(pokemon_list)
    }

@app.get("/config")
def get_config():
    """Get available configuration options"""
    return {
        "attributes": AVAILABLE_ATTRIBUTES,
        "algorithms": AVAILABLE_ALGORITHMS,
        "algorithm_descriptions": ALGORITHM_DESCRIPTIONS,
        "csp_heuristics": {
            "variable_ordering": {
                "options": VARIABLE_ORDERING_HEURISTICS,
                "descriptions": VARIABLE_HEURISTIC_DESCRIPTIONS
            },
            "value_ordering": {
                "options": VALUE_ORDERING_HEURISTICS,
                "descriptions": VALUE_HEURISTIC_DESCRIPTIONS
            }
        },
        "default_configs": {
            "csp": DEFAULT_CSP_CONFIG,
            "ga": DEFAULT_GA_CONFIG,
            "sa": DEFAULT_SA_CONFIG,
            "astar": DEFAULT_ASTAR_CONFIG
        }
    }

@app.get("/algorithms/{algorithm}")
def get_algorithm_info(algorithm: str):
    """Get detailed information about a specific algorithm"""
    if algorithm.upper() not in AVAILABLE_ALGORITHMS:
        raise HTTPException(404, f"Algorithm {algorithm} not found")
    
    algo = algorithm.upper()
    
    info = {
        "name": algo,
        "description": ALGORITHM_DESCRIPTIONS.get(algo),
        "config_options": {}
    }
    
    if algo == 'CSP':
        info["config_options"] = {
            "variable_heuristic": {
                "description": "Which attribute to constrain next",
                "options": VARIABLE_ORDERING_HEURISTICS,
                "details": VARIABLE_HEURISTIC_DESCRIPTIONS
            },
            "value_heuristic": {
                "description": "Which value to try for chosen attribute",
                "options": VALUE_ORDERING_HEURISTICS,
                "details": VALUE_HEURISTIC_DESCRIPTIONS
            },
            "use_ac3": {
                "description": "Use AC-3 constraint propagation",
                "type": "boolean",
                "default": True
            }
        }
        info["theoretical_properties"] = [
            "Complete: Guaranteed to find solution if one exists",
            "Optimal: Finds solution with proper heuristics",
            "Systematic: Explores search space systematically"
        ]
    
    elif algo == 'GA':
        info["config_options"] = DEFAULT_GA_CONFIG
        info["theoretical_properties"] = [
            "Stochastic: Uses randomness in selection and mutation",
            "Population-based: Maintains diverse candidate set",
            "Valid individuals: All Pokemon are real (not arbitrary combinations)"
        ]
    
    elif algo == 'ASTAR':
        info["config_options"] = DEFAULT_ASTAR_CONFIG
        info["theoretical_properties"] = [
            "Complete: Guaranteed to find solution",
            "Optimal: Finds shortest path with admissible heuristic",
            "Informed: Uses heuristic to guide search efficiently"
        ]
    
    elif algo == 'SA':
        info["config_options"] = DEFAULT_SA_CONFIG
        info["theoretical_properties"] = [
            "Probabilistic: Accepts worse solutions with decreasing probability",
            "Local search: Explores neighborhood of current solution",
            "Annealing: Temperature controls exploration vs exploitation"
        ]
    
    return info

@app.post("/solve")
def solve(config: SolverConfig):
    """
    Main solving endpoint with corrected algorithms.
    """
    start_time = time.time()
    
    # Validate configuration
    try:
        validate_config(config.dict())
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(400, str(e))
    
    # Get secret Pokemon
    if config.secret_pokemon:
        secret = data_loader.get_pokemon_by_name(config.secret_pokemon)
        if secret is None:
            raise HTTPException(400, f"Pokemon '{config.secret_pokemon}' not found")
    else:
        secret = data_loader.get_random_pokemon()
    
    # Create solver with corrected implementation
    try:
        solver = create_solver(config)
    except Exception as e:
        raise HTTPException(500, f"Failed to create solver: {str(e)}")
    
    # Solving loop
    steps = []
    success = False
    
    for attempt in range(1, config.max_attempts + 1):
        # Get next guess
        try:
            guess, algorithm_state = solver.next_guess()
        except Exception as e:
            raise HTTPException(500, f"Solver error at attempt {attempt}: {str(e)}")
        
        if guess is None:
            break
        
        # Calculate feedback with corrected logic
        feedback = get_feedback(secret, guess, config.attributes, NUMERIC_ATTRIBUTES)
        
        # Create step with algorithm state
        step = SolverStep(
            attempt=attempt,
            guess_name=guess['Original_Name'],
            guess_data={attr: str(guess.get(attr, 'N/A')) for attr in config.attributes},
            feedback=feedback,
            remaining_candidates=algorithm_state.get('candidates', 0),
            timestamp=time.time() - start_time,
            image_url=guess.get('image_url', ''),
            algorithm_state=algorithm_state
        )
        steps.append(step)
        
        # Check if solved
        if is_complete_match(feedback):
            success = True
            break
        
        # Update solver with feedback
        try:
            solver.update_feedback(guess, feedback)
        except Exception as e:
            raise HTTPException(500, f"Failed to update solver: {str(e)}")
    
    execution_time = time.time() - start_time
    
    # Calculate performance metrics
    metrics = calculate_metrics(steps, execution_time, success)
    
    # Build algorithm config for response
    algorithm_config = {}
    if config.algorithm == 'CSP':
        csp_conf = config.csp_config or CSPConfig()
        algorithm_config = {
            "variable_heuristic": csp_conf.variable_heuristic,
            "value_heuristic": csp_conf.value_heuristic,
            "use_ac3": csp_conf.use_ac3
        }
    elif config.algorithm == 'GA':
        algorithm_config = config.ga_config.dict() if config.ga_config else DEFAULT_GA_CONFIG
    elif config.algorithm == 'ASTAR':
        algorithm_config = config.astar_config.dict() if config.astar_config else DEFAULT_ASTAR_CONFIG
    elif config.algorithm == 'SA':
        algorithm_config = config.sa_config.dict() if config.sa_config else DEFAULT_SA_CONFIG
    
    return SolverResult(
        secret_name=secret['Original_Name'],
        secret_image=secret.get('image_url', ''),
        success=success,
        total_attempts=len(steps),
        steps=steps,
        execution_time=round(execution_time, 3),
        algorithm=config.algorithm,
        algorithm_config=algorithm_config,
        performance_metrics=metrics.to_dict()
    )

@app.post("/compare")
def compare_algorithms(
    algorithms: List[str],
    attributes: List[str],
    secret_pokemon: Optional[str] = None,
    max_attempts: int = 10
):
    """
    Compare multiple algorithms on the same Pokemon.
    Uses default configurations for fair comparison.
    """
    
    results = {}
    
    # Get secret Pokemon once
    if secret_pokemon:
        secret = data_loader.get_pokemon_by_name(secret_pokemon)
        if secret is None:
            raise HTTPException(400, f"Pokemon '{secret_pokemon}' not found")
    else:
        secret = data_loader.get_random_pokemon()
    
    secret_name = secret['Original_Name']
    
    for algo in algorithms:
        if algo.upper() not in AVAILABLE_ALGORITHMS:
            continue
        
        try:
            # Create config for this algorithm with defaults
            config = SolverConfig(
                algorithm=algo.upper(),
                attributes=attributes,
                secret_pokemon=secret_name,
                max_attempts=max_attempts
            )
            
            # Add algorithm-specific configs with defaults
            if algo.upper() == 'CSP':
                config.csp_config = CSPConfig()
            elif algo.upper() == 'GA':
                config.ga_config = GAConfig()
            elif algo.upper() == 'SA':
                config.sa_config = SAConfig()
            elif algo.upper() == 'ASTAR':
                config.astar_config = AStarConfig()
            
            result = solve(config)
            results[algo] = {
                "success": result.success,
                "attempts": result.total_attempts,
                "time": result.execution_time,
                "metrics": result.performance_metrics,
                "config": result.algorithm_config
            }
        except Exception as e:
            results[algo] = {"error": str(e)}
    
    # Determine winner (fewest attempts among successful runs)
    winner = None
    if results:
        valid_results = [(k, v) for k, v in results.items() 
                        if "error" not in v and v.get("success")]
        if valid_results:
            winner = min(valid_results, key=lambda x: x[1]["attempts"])[0]
    
    return {
        "secret_pokemon": secret_name,
        "results": results,
        "winner": winner,
        "comparison_notes": [
            "All algorithms use default configurations",
            "CSP uses MRV + LCV with AC-3",
            "GA uses valid Pokemon crossover",
            "A* uses admissible heuristic",
            "SA uses Metropolis criterion"
        ]
    }

@app.post("/test/csp-heuristics")
def test_csp_heuristics(
    attributes: List[str],
    secret_pokemon: Optional[str] = None,
    max_attempts: int = 10
):
    """
    Test different CSP heuristic combinations.
    """
    
    # Get secret Pokemon
    if secret_pokemon:
        secret = data_loader.get_pokemon_by_name(secret_pokemon)
        if secret is None:
            raise HTTPException(400, f"Pokemon '{secret_pokemon}' not found")
    else:
        secret = data_loader.get_random_pokemon()
    
    secret_name = secret['Original_Name']
    results = {}
    
    # Test different variable ordering heuristics
    for var_h in VARIABLE_ORDERING_HEURISTICS:
        for val_h in VALUE_ORDERING_HEURISTICS:
            key = f"{var_h}+{val_h}"
            
            try:
                config = SolverConfig(
                    algorithm='CSP',
                    attributes=attributes,
                    secret_pokemon=secret_name,
                    max_attempts=max_attempts,
                    csp_config=CSPConfig(
                        variable_heuristic=var_h,
                        value_heuristic=val_h,
                        use_ac3=True
                    )
                )
                
                result = solve(config)
                results[key] = {
                    "success": result.success,
                    "attempts": result.total_attempts,
                    "time": result.execution_time,
                    "variable_heuristic": var_h,
                    "value_heuristic": val_h
                }
            except Exception as e:
                results[key] = {"error": str(e)}
    
    # Find best combination
    valid_results = [(k, v) for k, v in results.items() 
                    if "error" not in v and v.get("success")]
    
    best_combo = None
    if valid_results:
        best_combo = min(valid_results, key=lambda x: x[1]["attempts"])[0]
    
    return {
        "secret_pokemon": secret_name,
        "results": results,
        "best_combination": best_combo,
        "note": "Tests all combinations of variable and value ordering heuristics"
    }
    
@app.post("/solve/stream")
async def solve_stream(config: SolverConfig):
    """Streaming endpoint with real-time updates SSE FORMAT"""
    
    async def event_generator():
        # Use a regular list instead of asyncio.Queue for synchronous callback
        progress_events = []
        
        def progress_callback(data):
            # Simply append to list - this works from sync context
            progress_events.append(data)
            print(f"[CALLBACK] Progress event added: gen={data.get('generation')}, fitness={data.get('best_fitness')}")
        
        try:
            validate_config(config.dict())
            
            # Get secret
            if config.secret_pokemon:
                secret = data_loader.get_pokemon_by_name(config.secret_pokemon)
                if not secret:
                    yield f"event: error\ndata: {json.dumps({'error': 'Pokemon not found'})}\n\n"
                    return
            else:
                secret = data_loader.get_random_pokemon()
            
            # Start event - PROPER SSE FORMAT
            yield f"event: start\ndata: {json.dumps({'secret_name': secret['Original_Name']})}\n\n"
            
            # Create solver
            df = data_loader.get_dataframe()
            
            # Create appropriate solver based on algorithm
            if config.algorithm == 'GA':
                ga_config = config.ga_config or GAConfig()
                solver = GASolver(df, config.attributes, ga_config.dict(), progress_callback)
            else:
                # For other algorithms without progress callback support
                solver = create_solver(config)
            
            # Solve
            start_time = time.time()
            steps = []
            success = False
            
            for attempt in range(1, config.max_attempts + 1):
                yield f"event: attempt_start\ndata: {json.dumps({'attempt': attempt})}\n\n"
                
                guess, algorithm_state = solver.next_guess()
                if guess is None:
                    break
                
                # Yield any accumulated progress events
                if progress_events:
                    print(f"[STREAM] Yielding {len(progress_events)} progress events")
                    for progress_data in progress_events:
                        yield f"event: progress\ndata: {json.dumps(progress_data)}\n\n"
                    progress_events.clear()  # Clear after yielding
                
                feedback = get_feedback(secret, guess, config.attributes, NUMERIC_ATTRIBUTES)
                
                step = {
                    "attempt": attempt,
                    "guess_name": guess['Original_Name'],
                    "guess_data": {attr: str(guess.get(attr, 'N/A')) for attr in config.attributes},
                    "feedback": feedback,
                    "algorithm_state": algorithm_state,
                    "image_url": guess.get('image_url', '')
                }
                steps.append(step)
                
                yield f"event: step\ndata: {json.dumps(step)}\n\n"
                
                if is_complete_match(feedback):
                    success = True
                    break
                
                solver.update_feedback(guess, feedback)
            
            execution_time = time.time() - start_time
            metrics = calculate_metrics(steps, execution_time, success)
            
            # PROPER SSE FORMAT for completion
            yield f"event: complete\ndata: {json.dumps({'success': success, 'total_attempts': len(steps), 'execution_time': round(execution_time, 3), 'performance_metrics': metrics.to_dict()})}\n\n"
            
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    
    return EventSourceResponse(event_generator())


@app.get("/algorithm-theory/{algorithm}")
def get_algorithm_theory(algorithm: str):
    """
    Get theoretical background and correctness properties of an algorithm.
    """
    if algorithm.upper() not in AVAILABLE_ALGORITHMS:
        raise HTTPException(404, f"Algorithm {algorithm} not found")
    
    algo = algorithm.upper()
    
    theories = {
        "CSP": {
            "formulation": {
                "variables": "Attributes to determine (Type1, Type2, Height, etc.)",
                "domains": "Possible values for each attribute",
                "constraints": "Rules derived from feedback",
                "solution": "Complete assignment satisfying all constraints"
            },
            "algorithms": {
                "AC-3": "Arc Consistency Algorithm #3 - propagates constraints to reduce domains",
                "Backtracking": "Systematic search with constraint checking",
                "Heuristics": "Guide search to reduce branching factor"
            },
            "properties": {
                "completeness": "Yes - finds solution if one exists",
                "optimality": "Depends on heuristic choice",
                "time_complexity": "O(d^n) worst case, much better with heuristics",
                "space_complexity": "O(n) for backtracking"
            },
            "correctness": [
                "Variables correctly represent attributes, not Pokemon",
                "AC-3 maintains arc consistency",
                "Two-level heuristics (variable + value ordering)",
                "Constraints properly model feedback"
            ]
        },
        "GA": {
            "formulation": {
                "individual": "A Pokemon (represented by index)",
                "population": "Set of candidate Pokemon",
                "fitness": "Constraint satisfaction score",
                "gene": "Pokemon index (immutable)",
                "selection": "Tournament selection based on fitness"
            },
            "operators": {
                "crossover": "Find real Pokemon matching parent attributes",
                "mutation": "Replace with similar Pokemon",
                "elitism": "Preserve best individuals"
            },
            "properties": {
                "completeness": "No - stochastic search",
                "optimality": "No - local optima possible",
                "time_complexity": "O(g * p * f) where g=generations, p=population, f=fitness eval",
                "space_complexity": "O(p) for population"
            },
            "correctness": [
                "Individuals are always valid Pokemon",
                "Crossover maintains validity (no arbitrary combinations)",
                "Fitness measures constraint satisfaction",
                "Diversity maintenance prevents premature convergence"
            ]
        },
        "ASTAR": {
            "formulation": {
                "state": "A Pokemon guess",
                "goal": "Secret Pokemon (unknown initially)",
                "g(n)": "Number of guesses so far (path cost)",
                "h(n)": "Estimated remaining guesses (heuristic)",
                "f(n)": "g(n) + h(n) - total estimated cost"
            },
            "algorithm": {
                "open_set": "Priority queue ordered by f(n)",
                "closed_set": "Already explored states",
                "search": "Best-first search with admissible heuristic"
            },
            "properties": {
                "completeness": "Yes - if heuristic is admissible",
                "optimality": "Yes - if heuristic is admissible and consistent",
                "time_complexity": "O(b^d) where b=branching, d=depth",
                "space_complexity": "O(b^d) for open/closed sets"
            },
            "correctness": [
                "Heuristic is admissible (never overestimates)",
                "Properly tracks path cost",
                "Uses closed set to avoid cycles",
                "Updates estimates with new feedback"
            ]
        },
        "SA": {
            "formulation": {
                "state": "A Pokemon candidate",
                "energy": "Constraint violation count (lower = better)",
                "neighbor": "Similar Pokemon",
                "temperature": "Controls exploration vs exploitation"
            },
            "algorithm": {
                "acceptance": "Metropolis criterion - exp(-ΔE/T)",
                "cooling": "Gradual temperature reduction",
                "reheating": "Optional restart with higher temperature"
            },
            "properties": {
                "completeness": "No - local search",
                "optimality": "Probabilistically optimal with slow cooling",
                "time_complexity": "O(i * n) where i=iterations, n=neighbor eval",
                "space_complexity": "O(1) - only current state"
            },
            "correctness": [
                "Energy function counts constraint violations",
                "Acceptance probability follows Metropolis criterion",
                "Temperature schedule controls annealing",
                "Neighbor generation maintains validity"
            ]
        }
    }
    
    return theories.get(algo, {"error": "Theory not available"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)