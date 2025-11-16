from typing import List, Dict, Any, Optional
from fastapi import HTTPException
from config import (
    AVAILABLE_ALGORITHMS,
    AVAILABLE_ATTRIBUTES,
    AVAILABLE_HEURISTICS,
    AVAILABLE_CROSSOVER_STRATEGIES
)

def validate_algorithm(algorithm: str) -> str:
    """
    Validate algorithm choice.
    
    Args:
        algorithm: Algorithm name to validate
        
    Returns:
        Validated algorithm name (uppercase)
        
    Raises:
        HTTPException: If algorithm is invalid
    """
    algorithm = algorithm.upper()
    if algorithm not in AVAILABLE_ALGORITHMS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid algorithm '{algorithm}'. Must be one of: {', '.join(AVAILABLE_ALGORITHMS)}"
        )
    return algorithm

def validate_attributes(attributes: List[str]) -> List[str]:
    """
    Validate attribute selection.
    
    Args:
        attributes: List of attribute names
        
    Returns:
        Validated attribute list
        
    Raises:
        HTTPException: If any attribute is invalid
    """
    if not attributes:
        raise HTTPException(
            status_code=400,
            detail="At least one attribute must be specified"
        )
    
    invalid_attrs = [attr for attr in attributes if attr not in AVAILABLE_ATTRIBUTES]
    if invalid_attrs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid attributes: {', '.join(invalid_attrs)}. Available: {', '.join(AVAILABLE_ATTRIBUTES)}"
        )
    
    return attributes

def validate_heuristic(heuristic: str, algorithm: str) -> str:
    """
    Validate heuristic choice for given algorithm.
    
    Args:
        heuristic: Heuristic name
        algorithm: Algorithm name
        
    Returns:
        Validated heuristic name
        
    Raises:
        HTTPException: If heuristic is invalid or incompatible with algorithm
    """
    if heuristic not in AVAILABLE_HEURISTICS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid heuristic '{heuristic}'. Must be one of: {', '.join(AVAILABLE_HEURISTICS)}"
        )
    
    # CSP can use all heuristics
    if algorithm == 'CSP':
        return heuristic
    
    # Other algorithms have limited heuristic support
    if algorithm in ['GA', 'ASTAR', 'SA'] and heuristic != 'random':
        # These algorithms use their own internal heuristics
        # But we allow the parameter for consistency
        pass
    
    return heuristic

def validate_crossover_strategy(strategy: str) -> str:
    """
    Validate crossover strategy for GA.
    
    Args:
        strategy: Crossover strategy name
        
    Returns:
        Validated strategy name
        
    Raises:
        HTTPException: If strategy is invalid
    """
    if strategy not in AVAILABLE_CROSSOVER_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid crossover strategy '{strategy}'. Must be one of: {', '.join(AVAILABLE_CROSSOVER_STRATEGIES)}"
        )
    return strategy

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate entire solver configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration
        
    Raises:
        HTTPException: If any part of config is invalid
    """
    # Validate required fields
    if 'algorithm' not in config:
        raise HTTPException(status_code=400, detail="Algorithm must be specified")
    
    if 'attributes' not in config:
        raise HTTPException(status_code=400, detail="Attributes must be specified")
    
    # Validate algorithm
    config['algorithm'] = validate_algorithm(config['algorithm'])
    
    # Validate attributes
    config['attributes'] = validate_attributes(config['attributes'])
    
    # Validate heuristic if provided
    if 'heuristic' in config:
        config['heuristic'] = validate_heuristic(config['heuristic'], config['algorithm'])
    
    # Validate max_attempts
    if 'max_attempts' in config:
        max_attempts = config['max_attempts']
        if not isinstance(max_attempts, int) or max_attempts < 1 or max_attempts > 50:
            raise HTTPException(
                status_code=400,
                detail="max_attempts must be an integer between 1 and 50"
            )
    
    # Validate GA config if present
    if config['algorithm'] == 'GA' and 'ga_config' in config:
        validate_ga_config(config['ga_config'])
    
    # Validate SA config if present
    if config['algorithm'] == 'SA' and 'sa_config' in config:
        validate_sa_config(config['sa_config'])
    
    # Validate A* config if present
    if config['algorithm'] == 'ASTAR' and 'astar_config' in config:
        validate_astar_config(config['astar_config'])
    
    return config

def validate_ga_config(ga_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate GA configuration parameters.
    
    Args:
        ga_config: GA configuration dictionary
        
    Returns:
        Validated GA config
        
    Raises:
        HTTPException: If any parameter is invalid
    """
    if 'pop_size' in ga_config:
        if not 10 <= ga_config['pop_size'] <= 500:
            raise HTTPException(
                status_code=400,
                detail="pop_size must be between 10 and 500"
            )
    
    if 'elite_size' in ga_config:
        if not 5 <= ga_config['elite_size'] <= 100:
            raise HTTPException(
                status_code=400,
                detail="elite_size must be between 5 and 100"
            )
        
        # Elite size should be less than population size
        if 'pop_size' in ga_config and ga_config['elite_size'] >= ga_config['pop_size']:
            raise HTTPException(
                status_code=400,
                detail="elite_size must be less than pop_size"
            )
    
    if 'mutation_rate' in ga_config:
        if not 0.0 <= ga_config['mutation_rate'] <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="mutation_rate must be between 0.0 and 1.0"
            )
    
    if 'crossover_rate' in ga_config:
        if not 0.0 <= ga_config['crossover_rate'] <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="crossover_rate must be between 0.0 and 1.0"
            )
    
    if 'tournament_size' in ga_config:
        if not 2 <= ga_config['tournament_size'] <= 20:
            raise HTTPException(
                status_code=400,
                detail="tournament_size must be between 2 and 20"
            )
    
    if 'crossover_strategy' in ga_config:
        validate_crossover_strategy(ga_config['crossover_strategy'])
    
    if 'generations_per_guess' in ga_config:
        if not 1 <= ga_config['generations_per_guess'] <= 200:
            raise HTTPException(
                status_code=400,
                detail="generations_per_guess must be between 1 and 200"
            )
    
    return ga_config

def validate_sa_config(sa_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate Simulated Annealing configuration.
    
    Args:
        sa_config: SA configuration dictionary
        
    Returns:
        Validated SA config
        
    Raises:
        HTTPException: If any parameter is invalid
    """
    if 'initial_temp' in sa_config:
        if sa_config['initial_temp'] <= 0:
            raise HTTPException(
                status_code=400,
                detail="initial_temp must be greater than 0"
            )
    
    if 'cooling_rate' in sa_config:
        if not 0.0 < sa_config['cooling_rate'] < 1.0:
            raise HTTPException(
                status_code=400,
                detail="cooling_rate must be between 0.0 and 1.0"
            )
    
    if 'min_temp' in sa_config:
        if sa_config['min_temp'] <= 0:
            raise HTTPException(
                status_code=400,
                detail="min_temp must be greater than 0"
            )
        
        # Min temp should be less than initial temp
        if 'initial_temp' in sa_config and sa_config['min_temp'] >= sa_config['initial_temp']:
            raise HTTPException(
                status_code=400,
                detail="min_temp must be less than initial_temp"
            )
    
    if 'iterations_per_temp' in sa_config:
        if sa_config['iterations_per_temp'] < 1:
            raise HTTPException(
                status_code=400,
                detail="iterations_per_temp must be at least 1"
            )
    
    if 'reheat_threshold' in sa_config:
        if not 0.0 <= sa_config['reheat_threshold'] <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="reheat_threshold must be between 0.0 and 1.0"
            )
    
    return sa_config

def validate_astar_config(astar_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate A* configuration.
    
    Args:
        astar_config: A* configuration dictionary
        
    Returns:
        Validated A* config
        
    Raises:
        HTTPException: If any parameter is invalid
    """
    if 'max_open_set' in astar_config:
        if astar_config['max_open_set'] < 10:
            raise HTTPException(
                status_code=400,
                detail="max_open_set must be at least 10"
            )
    
    if 'beam_width' in astar_config:
        if astar_config['beam_width'] < 1:
            raise HTTPException(
                status_code=400,
                detail="beam_width must be at least 1"
            )
    
    if 'heuristic_weight' in astar_config:
        if astar_config['heuristic_weight'] < 0:
            raise HTTPException(
                status_code=400,
                detail="heuristic_weight must be non-negative"
            )
    
    return astar_config

def validate_pokemon_name(name: str, available_pokemon: List[str]) -> str:
    """
    Validate Pokemon name.
    
    Args:
        name: Pokemon name to validate
        available_pokemon: List of available Pokemon names
        
    Returns:
        Validated Pokemon name
        
    Raises:
        HTTPException: If Pokemon not found
    """
    if name not in available_pokemon:
        raise HTTPException(
            status_code=404,
            detail=f"Pokemon '{name}' not found in dataset"
        )
    return name