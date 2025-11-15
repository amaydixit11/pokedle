import pandas as pd
import math
import random
from typing import Dict, Tuple, Any, List, Set
from algorithms.base import BaseSolver

class SimulatedAnnealingSolver(BaseSolver):
    """
    FIXED Simulated Annealing algorithm for Pokedle.
    
    Key fixes:
    1. Proper energy calculation (constraint violations)
    2. Better neighbor generation strategy
    3. Correct acceptance probability
    4. Improved constraint tracking from feedback
    5. Better handling of None/NaN values
    """
    
    def __init__(self, dataframe: pd.DataFrame, attributes: list, config: dict):
        super().__init__(dataframe, attributes)
        
        # SA parameters
        self.initial_temp = config.get('initial_temp', 100.0)
        self.cooling_rate = config.get('cooling_rate', 0.95)
        self.min_temp = config.get('min_temp', 0.01)
        self.iterations_per_temp = config.get('iterations_per_temp', 50)
        self.reheat_threshold = config.get('reheat_threshold', 0.1)
        
        # Current state
        self.current_temp = self.initial_temp
        self.current_solution = None
        self.best_solution = None
        self.best_energy = float('inf')
        self.iteration = 0
        self.no_improvement_count = 0
        
        # Constraint tracking - STRUCTURED
        self.constraints = {
            'must_equal': {},      # {attr: value}
            'not_equal': {},       # {attr: set of values}
            'type_constraints': {},  # Special handling for Type1/Type2
            'numeric_constraints': {}  # {attr: {'min': x, 'max': y}}
        }
        
        # Valid candidates based on constraints
        self.valid_candidates = set(dataframe.index)
    
    def _safe_get_value(self, pokemon, attr: str):
        """Safely get attribute value, handling None/NaN"""
        if isinstance(pokemon, pd.Series):
            val = pokemon.get(attr)
        else:
            val = pokemon[attr]
        
        if pd.isna(val):
            return None
        return val
    
    def _get_pokemon_types(self, pokemon) -> Set:
        """Get Pokemon types as a set, excluding None"""
        types = set()
        type1 = self._safe_get_value(pokemon, 'Type1')
        type2 = self._safe_get_value(pokemon, 'Type2')
        
        if type1 is not None:
            types.add(type1)
        if type2 is not None:
            types.add(type2)
        
        return types
    
    def update_constraints_from_feedback(self, guess: pd.Series, feedback: Dict[str, str]):
        """
        FIXED: Properly update constraints from feedback.
        """
        for attr, status in feedback.items():
            if attr not in self.attributes or attr == 'image_url':
                continue
            
            guess_val = self._safe_get_value(guess, attr)
            
            if status == 'green':
                # MUST equal this value
                self.constraints['must_equal'][attr] = guess_val
                
            elif status == 'gray':
                # Must NOT equal this value
                if attr in ['Type1', 'Type2']:
                    # Type doesn't appear anywhere
                    if 'types_excluded' not in self.constraints['type_constraints']:
                        self.constraints['type_constraints']['types_excluded'] = set()
                    if guess_val is not None:
                        self.constraints['type_constraints']['types_excluded'].add(guess_val)
                else:
                    if attr not in self.constraints['not_equal']:
                        self.constraints['not_equal'][attr] = set()
                    if guess_val is not None:
                        self.constraints['not_equal'][attr].add(guess_val)
            
            elif status == 'yellow':
                # Type exists but in wrong position
                if attr in ['Type1', 'Type2']:
                    other_attr = 'Type2' if attr == 'Type1' else 'Type1'
                    
                    # This type must exist
                    if 'types_required' not in self.constraints['type_constraints']:
                        self.constraints['type_constraints']['types_required'] = set()
                    if guess_val is not None:
                        self.constraints['type_constraints']['types_required'].add(guess_val)
                    
                    # But not in this position
                    if attr not in self.constraints['not_equal']:
                        self.constraints['not_equal'][attr] = set()
                    if guess_val is not None:
                        self.constraints['not_equal'][attr].add(guess_val)
            
            elif status == 'higher':
                # Value must be GREATER than guess_val
                if attr not in self.constraints['numeric_constraints']:
                    self.constraints['numeric_constraints'][attr] = {}
                if guess_val is not None:
                    self.constraints['numeric_constraints'][attr]['min'] = float(guess_val)
            
            elif status == 'lower':
                # Value must be LESS than guess_val
                if attr not in self.constraints['numeric_constraints']:
                    self.constraints['numeric_constraints'][attr] = {}
                if guess_val is not None:
                    self.constraints['numeric_constraints'][attr]['max'] = float(guess_val)
        
        # Update valid candidates
        self._update_valid_candidates()
    
    def _update_valid_candidates(self):
        """Update set of valid candidates based on constraints"""
        valid = set()
        
        for idx in self.df.index:
            if self._satisfies_all_constraints(idx):
                valid.add(idx)
        
        self.valid_candidates = valid
    
    def _satisfies_all_constraints(self, pokemon_idx: int) -> bool:
        """Check if Pokemon satisfies all hard constraints"""
        pokemon = self.df.loc[pokemon_idx]
        
        # Check must_equal constraints
        for attr, required_val in self.constraints['must_equal'].items():
            pokemon_val = self._safe_get_value(pokemon, attr)
            if pokemon_val != required_val:
                return False
        
        # Check not_equal constraints
        for attr, excluded_vals in self.constraints['not_equal'].items():
            pokemon_val = self._safe_get_value(pokemon, attr)
            if pokemon_val in excluded_vals:
                return False
        
        # Check type constraints
        if self.constraints['type_constraints']:
            pokemon_types = self._get_pokemon_types(pokemon)
            
            # Required types must exist
            required_types = self.constraints['type_constraints'].get('types_required', set())
            if not required_types.issubset(pokemon_types):
                return False
            
            # Excluded types must not exist
            excluded_types = self.constraints['type_constraints'].get('types_excluded', set())
            if pokemon_types.intersection(excluded_types):
                return False
        
        # Check numeric constraints
        for attr, bounds in self.constraints['numeric_constraints'].items():
            pokemon_val = self._safe_get_value(pokemon, attr)
            if pokemon_val is None:
                return False
            
            try:
                val_float = float(pokemon_val)
                
                if 'min' in bounds and val_float <= bounds['min']:
                    return False
                
                if 'max' in bounds and val_float >= bounds['max']:
                    return False
            except (ValueError, TypeError):
                return False
        
        return True
    
    def energy(self, pokemon_idx: int) -> float:
        """
        FIXED: Calculate energy (constraint violations + exploration bonus).
        Lower energy = better solution.
        
        CRITICAL FIX: Add small exploration bonus to prevent getting stuck
        at local optima (Pokemon that satisfy all constraints but aren't the answer).
        """
        pokemon = self.df.loc[pokemon_idx]
        violations = 0.0
        
        # If no feedback yet, use diversity heuristic
        if not self.feedback_history:
            return self._diversity_energy(pokemon)
        
        # Count violations for each feedback
        for guess_idx, feedback in self.feedback_history:
            guess = self.df.loc[guess_idx]
            
            for attr, status in feedback.items():
                if attr not in self.attributes or attr == 'image_url':
                    continue
                
                pokemon_val = self._safe_get_value(pokemon, attr)
                guess_val = self._safe_get_value(guess, attr)
                
                if status == 'green':
                    # CRITICAL: Must match exactly
                    if pokemon_val != guess_val:
                        violations += 10.0  # Heavy penalty
                
                elif status == 'gray':
                    if attr in ['Type1', 'Type2']:
                        # Type must not appear anywhere
                        pokemon_types = self._get_pokemon_types(pokemon)
                        if guess_val in pokemon_types:
                            violations += 5.0
                    else:
                        # Value must not match
                        if pokemon_val == guess_val:
                            violations += 5.0
                
                elif status == 'yellow':
                    pokemon_types = self._get_pokemon_types(pokemon)
                    
                    # Type must exist somewhere
                    if guess_val not in pokemon_types:
                        violations += 5.0
                    
                    # But not in this exact position
                    if pokemon_val == guess_val:
                        violations += 2.0
                
                elif status == 'higher':
                    # Pokemon value must be GREATER
                    try:
                        if pokemon_val is not None and guess_val is not None:
                            if float(pokemon_val) <= float(guess_val):
                                violations += 5.0
                    except (ValueError, TypeError):
                        violations += 5.0
                
                elif status == 'lower':
                    # Pokemon value must be LESS
                    try:
                        if pokemon_val is not None and guess_val is not None:
                            if float(pokemon_val) >= float(guess_val):
                                violations += 5.0
                    except (ValueError, TypeError):
                        violations += 5.0
        
        # CRITICAL FIX: Add small diversity bonus to prevent getting stuck
        # This encourages exploration even when constraints are satisfied
        if violations == 0 and len(self.valid_candidates) > 1:
            # Small penalty based on how common this Pokemon's attributes are
            diversity_penalty = self._diversity_energy(pokemon) * 0.1
            violations += diversity_penalty
        
        return violations
    
    def _diversity_energy(self, pokemon: pd.Series) -> float:
        """Energy based on attribute commonness (for initial exploration)"""
        energy = 0.0
        
        for attr in self.attributes:
            if attr == 'image_url':
                continue
            
            value = self._safe_get_value(pokemon, attr)
            if value is None:
                energy += 0.5
                continue
            
            # More common = higher energy (prefer rare values for exploration)
            frequency = (self.df[attr] == value).sum() / len(self.df)
            energy += frequency
        
        return energy
    
    def acceptance_probability(self, current_energy: float, new_energy: float) -> float:
        """
        Metropolis acceptance criterion.
        Always accept better solutions, sometimes accept worse.
        """
        if new_energy < current_energy:
            return 1.0
        
        if self.current_temp <= 0:
            return 0.0
        
        # Metropolis criterion: exp(-ΔE/T)
        delta_energy = new_energy - current_energy
        probability = math.exp(-delta_energy / self.current_temp)
        
        return probability
    
    def get_neighbor(self, pokemon_idx: int) -> int:
        """
        FIXED: Generate neighbor solution intelligently.
        
        Strategy:
        - High temperature: Random exploration
        - Low temperature: Similar Pokemon from valid candidates
        """
        # If we have valid candidates, sample from them
        if self.valid_candidates:
            candidates_list = list(self.valid_candidates)
            
            # High temperature: more random
            if self.current_temp > self.initial_temp * 0.5:
                return random.choice(candidates_list)
            
            # Low temperature: prefer similar Pokemon
            current_pokemon = self.df.loc[pokemon_idx]
            
            # Sample for efficiency
            sample_size = min(100, len(candidates_list))
            sample = random.sample(candidates_list, sample_size)
            
            # Calculate similarity scores
            similarities = []
            for candidate_idx in sample:
                candidate = self.df.loc[candidate_idx]
                similarity = 0
                
                for attr in self.attributes:
                    if attr == 'image_url':
                        continue
                    
                    curr_val = self._safe_get_value(current_pokemon, attr)
                    cand_val = self._safe_get_value(candidate, attr)
                    
                    if curr_val == cand_val:
                        similarity += 1
                    elif attr in ['Height', 'Weight'] and curr_val is not None and cand_val is not None:
                        # Numeric similarity
                        try:
                            diff = abs(float(curr_val) - float(cand_val))
                            max_diff = self.df[attr].max() - self.df[attr].min()
                            if max_diff > 0:
                                similarity += 1 - (diff / max_diff)
                        except (ValueError, TypeError):
                            pass
                
                similarities.append((candidate_idx, similarity))
            
            # Weighted random selection (prefer similar)
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_k = min(10, len(similarities))
            top_candidates = similarities[:top_k]
            
            weights = [s + 0.1 for _, s in top_candidates]  # Add small constant
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
            
            return random.choices([idx for idx, _ in top_candidates], weights=probs)[0]
        
        # Fallback: random Pokemon
        return random.randint(0, len(self.df) - 1)
    
    def next_guess(self) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        FIXED: Generate next guess using simulated annealing.
        
        KEY FIX: Each call returns a DIFFERENT Pokemon from valid candidates
        to ensure we explore the solution space properly.
        """
        # Initialize if first guess
        if self.current_solution is None:
            if self.valid_candidates:
                self.current_solution = random.choice(list(self.valid_candidates))
            else:
                self.current_solution = random.randint(0, len(self.df) - 1)
            
            self.best_solution = self.current_solution
            self.best_energy = self.energy(self.current_solution)
        
        # CRITICAL FIX: If we have multiple valid candidates, explore them
        # Don't just return the same Pokemon over and over
        if len(self.valid_candidates) > 1:
            # Remove previously guessed Pokemon from consideration
            unguessed_candidates = self.valid_candidates.copy()
            
            for guess_idx, _ in self.feedback_history:
                unguessed_candidates.discard(guess_idx)
            
            # If we still have unguessed valid candidates, pick from them
            if unguessed_candidates:
                # Run SA iterations to find best among unguessed
                candidates_list = list(unguessed_candidates)
                
                # Start from random unguessed candidate
                exploration_solution = random.choice(candidates_list)
                exploration_energy = self.energy(exploration_solution)
                
                # Run limited SA iterations
                temp = self.current_temp
                for _ in range(self.iterations_per_temp):
                    # Generate neighbor from unguessed candidates
                    neighbor = random.choice(candidates_list)
                    neighbor_energy = self.energy(neighbor)
                    
                    # Accept if better or probabilistically
                    if random.random() < self.acceptance_probability(exploration_energy, neighbor_energy):
                        exploration_solution = neighbor
                        exploration_energy = neighbor_energy
                    
                    self.iteration += 1
                
                # Use the best solution found in exploration
                self.current_solution = exploration_solution
                self.best_solution = exploration_solution
                self.best_energy = exploration_energy
                
                # Cool down
                self.current_temp *= self.cooling_rate
                if self.current_temp < self.min_temp:
                    self.current_temp = self.min_temp
                
                # Return this new candidate
                pokemon = self.df.loc[self.best_solution]
                
                info = {
                    "algorithm": "simulated_annealing",
                    "temperature": round(self.current_temp, 3),
                    "current_energy": round(exploration_energy, 3),
                    "best_energy": round(self.best_energy, 3),
                    "iteration": self.iteration,
                    "no_improvement": 0,
                    "valid_candidates": len(self.valid_candidates),
                    "unguessed_candidates": len(unguessed_candidates),
                    "exploration_mode": True,
                    "constraints": {
                        "must_equal": len(self.constraints['must_equal']),
                        "not_equal": sum(len(v) for v in self.constraints['not_equal'].values()),
                        "type_constraints": len(self.constraints['type_constraints']),
                        "numeric": len(self.constraints['numeric_constraints'])
                    }
                }
                
                return pokemon, info
        
        # Standard SA if only one candidate or no valid candidates
        # Run iterations at current temperature
        for _ in range(self.iterations_per_temp):
            # Generate neighbor
            neighbor = self.get_neighbor(self.current_solution)
            
            # Calculate energies
            current_energy = self.energy(self.current_solution)
            neighbor_energy = self.energy(neighbor)
            
            # Acceptance decision
            if random.random() < self.acceptance_probability(current_energy, neighbor_energy):
                self.current_solution = neighbor
                
                # Update best solution
                if neighbor_energy < self.best_energy:
                    self.best_solution = neighbor
                    self.best_energy = neighbor_energy
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
            
            self.iteration += 1
            
            # Early stop if perfect solution found
            if self.best_energy == 0:
                break
        
        # Cool down
        self.current_temp *= self.cooling_rate
        
        # Reheating if stuck
        if self.no_improvement_count > 100:
            self.current_temp = self.initial_temp * self.reheat_threshold
            self.no_improvement_count = 0
        
        # Enforce minimum temperature
        if self.current_temp < self.min_temp:
            self.current_temp = self.min_temp
        
        # Return best solution found
        pokemon = self.df.loc[self.best_solution]
        
        info = {
            "algorithm": "simulated_annealing",
            "temperature": round(self.current_temp, 3),
            "current_energy": round(self.energy(self.current_solution), 3),
            "best_energy": round(self.best_energy, 3),
            "iteration": self.iteration,
            "no_improvement": self.no_improvement_count,
            "valid_candidates": len(self.valid_candidates),
            "exploration_mode": False,
            "constraints": {
                "must_equal": len(self.constraints['must_equal']),
                "not_equal": sum(len(v) for v in self.constraints['not_equal'].values()),
                "type_constraints": len(self.constraints['type_constraints']),
                "numeric": len(self.constraints['numeric_constraints'])
            }
        }
        
        return pokemon, info
    
    def update_feedback(self, guess: pd.Series, feedback: Dict[str, str]):
        """
        FIXED: Update solver with new feedback.
        """
        guess_idx = guess.name
        self.add_feedback(guess_idx, feedback)
        
        # Update constraints
        self.update_constraints_from_feedback(guess, feedback)
        
        # Re-evaluate current and best solutions
        if self.current_solution is not None:
            current_energy = self.energy(self.current_solution)
            best_energy = self.energy(self.best_solution)
            
            # If current solution violates new constraints, reheat
            if current_energy > self.best_energy + 10:
                self.current_temp = self.initial_temp * 0.7
                self.no_improvement_count = 0
                
                # Find new current solution from valid candidates
                if self.valid_candidates:
                    self.current_solution = random.choice(list(self.valid_candidates))
            
            # Update best if it's no longer valid
            if best_energy > 0 and self.valid_candidates:
                # Find best valid candidate
                best_idx = min(self.valid_candidates, key=lambda idx: self.energy(idx))
                new_energy = self.energy(best_idx)
                if new_energy < self.best_energy:
                    self.best_solution = best_idx
                    self.best_energy = new_energy
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information"""
        return {
            "algorithm": "SA",
            "temperature": round(self.current_temp, 3),
            "best_energy": round(self.best_energy, 3),
            "iteration": self.iteration,
            "valid_candidates": len(self.valid_candidates),
            "constraints": {
                "must_equal": len(self.constraints['must_equal']),
                "not_equal": sum(len(v) for v in self.constraints['not_equal'].values()),
                "type_constraints": len(self.constraints['type_constraints']),
                "numeric": len(self.constraints['numeric_constraints'])
            }
        }