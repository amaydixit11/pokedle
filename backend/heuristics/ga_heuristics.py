import pandas as pd
import random
from typing import Dict, List, Tuple, Any
from heuristics.base import BaseHeuristic

class GAHeuristics:
    """Collection of GA-specific heuristic functions"""
    
    @staticmethod
    def fitness_proportionate_selection(population: List[int], fitness_scores: List[float]) -> int:
        """
        Roulette wheel selection based on fitness.
        Higher fitness = higher selection probability.
        """
        if not fitness_scores or sum(fitness_scores) == 0:
            return random.choice(population)
        
        total_fitness = sum(fitness_scores)
        pick = random.uniform(0, total_fitness)
        current = 0
        
        for idx, fitness in zip(population, fitness_scores):
            current += fitness
            if current >= pick:
                return idx
        
        return population[-1]
    
    @staticmethod
    def rank_based_selection(population: List[int], fitness_scores: List[float]) -> int:
        """
        Rank-based selection to reduce selection pressure.
        Ranks individuals by fitness, selection based on rank.
        """
        if not population:
            return None
        
        # Create rank-fitness pairs
        ranked = sorted(zip(population, fitness_scores), key=lambda x: x[1])
        ranks = list(range(1, len(ranked) + 1))
        
        # Select based on rank probability
        total_rank = sum(ranks)
        pick = random.uniform(0, total_rank)
        current = 0
        
        for (idx, _), rank in zip(ranked, ranks):
            current += rank
            if current >= pick:
                return idx
        
        return ranked[-1][0]
    
    @staticmethod
    def stochastic_universal_sampling(population: List[int], fitness_scores: List[float], 
                                      n_select: int) -> List[int]:
        """
        Stochastic Universal Sampling for fairer selection.
        Ensures low-variance sampling.
        """
        if not fitness_scores or sum(fitness_scores) == 0:
            return random.sample(population, min(n_select, len(population)))
        
        total_fitness = sum(fitness_scores)
        point_distance = total_fitness / n_select
        start_point = random.uniform(0, point_distance)
        
        selected = []
        current_member = 0
        current_sum = fitness_scores[0]
        
        for i in range(n_select):
            pointer = start_point + i * point_distance
            
            while current_sum < pointer and current_member < len(population) - 1:
                current_member += 1
                current_sum += fitness_scores[current_member]
            
            selected.append(population[current_member])
        
        return selected
    
    @staticmethod
    def boltzmann_selection(population: List[int], fitness_scores: List[float], 
                           temperature: float = 1.0) -> int:
        """
        Boltzmann selection with temperature parameter.
        Higher temperature = more random selection.
        """
        if not fitness_scores:
            return random.choice(population)
        
        # Calculate Boltzmann probabilities
        import math
        boltzmann_scores = [math.exp(f / temperature) for f in fitness_scores]
        total = sum(boltzmann_scores)
        
        if total == 0:
            return random.choice(population)
        
        probs = [b / total for b in boltzmann_scores]
        
        # Select based on probabilities
        return random.choices(population, weights=probs)[0]
    
    @staticmethod
    def diversity_based_selection(population: List[int], df: pd.DataFrame, 
                                  attributes: List[str], n_select: int = 1) -> List[int]:
        """
        Select individuals that maximize population diversity.
        """
        if len(population) <= n_select:
            return population
        
        selected = []
        remaining = population.copy()
        
        # Select first individual randomly
        first = random.choice(remaining)
        selected.append(first)
        remaining.remove(first)
        
        # Iteratively select most diverse individuals
        while len(selected) < n_select and remaining:
            max_diversity = -1
            best_candidate = None
            
            for candidate in remaining:
                # Calculate diversity score
                diversity = 0
                candidate_pokemon = df.loc[candidate]
                
                for selected_idx in selected:
                    selected_pokemon = df.loc[selected_idx]
                    
                    # Count different attributes
                    for attr in attributes:
                        if attr == 'image_url':
                            continue
                        if candidate_pokemon[attr] != selected_pokemon[attr]:
                            diversity += 1
                
                if diversity > max_diversity:
                    max_diversity = diversity
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected
    
    @staticmethod
    def adaptive_mutation_rate(generation: int, max_generations: int, 
                              base_rate: float = 0.15, min_rate: float = 0.05) -> float:
        """
        Calculate adaptive mutation rate that decreases with generations.
        """
        progress = generation / max_generations if max_generations > 0 else 0
        return base_rate * (1 - progress) + min_rate * progress
    
    @staticmethod
    def fitness_sharing(population: List[int], fitness_scores: List[float], 
                       df: pd.DataFrame, attributes: List[str], 
                       sharing_radius: float = 0.3) -> List[float]:
        """
        Apply fitness sharing to maintain diversity.
        Similar individuals share fitness, reducing niche overcrowding.
        """
        shared_fitness = []
        
        for i, idx in enumerate(population):
            niche_count = 0
            pokemon_i = df.loc[idx]
            
            for j, other_idx in enumerate(population):
                pokemon_j = df.loc[other_idx]
                
                # Calculate similarity
                similarity = 0
                for attr in attributes:
                    if attr == 'image_url':
                        continue
                    if not pd.isna(pokemon_i[attr]) and not pd.isna(pokemon_j[attr]):
                        if pokemon_i[attr] == pokemon_j[attr]:
                            similarity += 1
                
                distance = 1 - (similarity / len(attributes))
                
                # Apply sharing function
                if distance < sharing_radius:
                    niche_count += 1 - (distance / sharing_radius)
            
            # Share fitness
            shared = fitness_scores[i] / max(niche_count, 1)
            shared_fitness.append(shared)
        
        return shared_fitness
    
    @staticmethod
    def crowding_distance(population: List[int], fitness_scores: List[float], 
                         df: pd.DataFrame, attributes: List[str]) -> Dict[int, float]:
        """
        Calculate crowding distance for each individual.
        Used in NSGA-II for diversity preservation.
        """
        distances = {idx: 0.0 for idx in population}
        
        if len(population) <= 2:
            for idx in population:
                distances[idx] = float('inf')
            return distances
        
        # For each attribute, calculate crowding distance
        for attr in attributes:
            if attr == 'image_url':
                continue
            
            # Sort population by attribute value
            sorted_pop = sorted(population, key=lambda idx: df.loc[idx][attr] 
                              if not pd.isna(df.loc[idx][attr]) else 0)
            
            # Boundary points have infinite distance
            distances[sorted_pop[0]] = float('inf')
            distances[sorted_pop[-1]] = float('inf')
            
            # Calculate distance for intermediate points
            attr_range = df[attr].max() - df[attr].min()
            if attr_range > 0:
                for i in range(1, len(sorted_pop) - 1):
                    prev_val = df.loc[sorted_pop[i-1]][attr]
                    next_val = df.loc[sorted_pop[i+1]][attr]
                    
                    if not pd.isna(prev_val) and not pd.isna(next_val):
                        distances[sorted_pop[i]] += (next_val - prev_val) / attr_range
        
        return distances