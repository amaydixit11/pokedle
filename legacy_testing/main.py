# main.py - Enhanced FastAPI Backend for Pokedle with Advanced GA
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import random
from typing import List, Optional, Dict
import time
import math

app = FastAPI(title="Pokedle Solver API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset
df = pd.read_csv("03_cleaned_with_images_and_evolutionary_stages.csv")

AVAILABLE_ATTRIBUTES = ['Generation', 'Height', 'Weight', 'Type1', 'Type2', 'Color', 'evolutionary_stage']
AVAILABLE_ALGORITHMS = ['CSP', 'GA']
AVAILABLE_HEURISTICS = ['random', 'mrv', 'lcv', 'entropy']
AVAILABLE_CROSSOVER_STRATEGIES = ['attribute_blend', 'uniform', 'single_point', 'fitness_weighted']

# ============ Models ============
class GAConfig(BaseModel):
    pop_size: int = 100
    elite_size: int = 20
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    tournament_size: int = 7
    crossover_strategy: str = 'attribute_blend'
    generations_per_guess: int = 30

class SolverConfig(BaseModel):
    algorithm: str
    attributes: List[str]
    heuristic: str = 'random'
    secret_pokemon: Optional[str] = None
    max_attempts: int = 10
    ga_config: Optional[GAConfig] = None

class SolverStep(BaseModel):
    attempt: int
    guess_name: str
    guess_data: Dict
    feedback: Dict
    remaining_candidates: int
    timestamp: float
    image_url: Optional[str] = None
    heuristic_info: Optional[Dict] = None

class SolverResult(BaseModel):
    secret_name: str
    secret_image: str
    success: bool
    total_attempts: int
    steps: List[SolverStep]
    execution_time: float

# ============ Feedback Function ============
def get_feedback(secret, guess, attributes, numeric_attrs=['Height', 'Weight']):
    feedback = {}
    
    secret_types = {secret.get('Type1'), secret.get('Type2')} - {None, float('nan')}
    secret_types = {t for t in secret_types if not (isinstance(t, float) and pd.isna(t))}
    
    guess_types = {guess.get('Type1'), guess.get('Type2')} - {None, float('nan')}
    guess_types = {t for t in guess_types if not (isinstance(t, float) and pd.isna(t))}
    
    for attr in attributes:
        if attr == 'image_url':
            continue
        
        if attr in ['Type1', 'Type2']:
            guess_value = guess[attr]
            secret_value = secret[attr]
            
            if pd.isna(guess_value) and pd.isna(secret_value):
                feedback[attr] = 'green'
            elif pd.isna(guess_value) or pd.isna(secret_value):
                feedback[attr] = 'gray'
            elif guess_value == secret_value:
                feedback[attr] = 'green'
            elif guess_value in secret_types:
                feedback[attr] = 'yellow'
            else:
                feedback[attr] = 'gray'
        elif pd.isna(secret[attr]) or pd.isna(guess[attr]):
            feedback[attr] = 'gray'
        elif secret[attr] == guess[attr]:
            feedback[attr] = 'green'
        elif attr in numeric_attrs:
            if guess[attr] < secret[attr]:
                feedback[attr] = 'higher'
            else:
                feedback[attr] = 'lower'
        else:
            feedback[attr] = 'gray'
    
    return feedback

# ============ Enhanced CSP Solver ============
class CSPSolver:
    def __init__(self, dataframe, attributes, heuristic='random', numeric_attrs=['Height', 'Weight']):
        self.df = dataframe.copy()
        self.attributes = attributes
        self.numeric_attrs = numeric_attrs
        self.heuristic = heuristic
        self.constraints = {col: [] for col in self.attributes}
        self.type_must_have = set()
        
    def apply_feedback(self, guess, feedback):
        for attr, status in feedback.items():
            if attr == 'image_url':
                continue
                
            value = guess[attr]
            
            if attr in ['Type1', 'Type2']:
                if status == "green":
                    self.constraints[attr].append(("==", value))
                elif status == "yellow":
                    self.constraints[attr].append(("!=", value))
                    if not pd.isna(value):
                        self.type_must_have.add(value)
                elif status == "gray":
                    self.constraints[attr].append(("!=", value))
            elif attr in self.numeric_attrs:
                if status == "green":
                    self.constraints[attr].append(("==", value))
            else:
                if status == "green":
                    self.constraints[attr].append(("==", value))
                elif status == "gray":
                    self.constraints[attr].append(("!=", value))
    
    def apply_numeric_feedback(self, attr, guess_value, secret_value):
        if attr not in self.numeric_attrs:
            return
            
        if guess_value == secret_value:
            self.constraints[attr].append(("==", secret_value))
        elif guess_value < secret_value:
            self.constraints[attr].append((">", guess_value))
        else:
            self.constraints[attr].append(("<", guess_value))
    
    def filter_candidates(self):
        candidates = self.df
        
        for attr, conds in self.constraints.items():
            if attr == 'image_url':
                continue
                
            for op, val in conds:
                if pd.isna(val):
                    continue
                    
                if op == "==":
                    candidates = candidates[candidates[attr] == val]
                elif op == "!=":
                    candidates = candidates[candidates[attr] != val]
                elif op == ">":
                    candidates = candidates[candidates[attr] > val]
                elif op == "<":
                    candidates = candidates[candidates[attr] < val]
        
        if self.type_must_have:
            def has_required_types(row):
                pokemon_types = {row['Type1'], row['Type2']} - {None, float('nan')}
                pokemon_types = {t for t in pokemon_types if not (isinstance(t, float) and pd.isna(t))}
                return self.type_must_have.issubset(pokemon_types)
            
            candidates = candidates[candidates.apply(has_required_types, axis=1)]
        
        return candidates
    
    def calculate_entropy(self, candidates, attr):
        if len(candidates) == 0:
            return 0
        
        value_counts = candidates[attr].value_counts()
        total = len(candidates)
        entropy = 0
        
        for count in value_counts:
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def next_guess_mrv(self, candidates):
        if len(candidates) == 0:
            return None, {}
        if len(candidates) == 1:
            return candidates.iloc[0], {"heuristic": "mrv", "candidates": 1}
        
        min_values = float('inf')
        best_attr = None
        
        for attr in self.attributes:
            if attr == 'image_url':
                continue
            unique_count = candidates[attr].nunique()
            if unique_count < min_values and unique_count > 0:
                min_values = unique_count
                best_attr = attr
        
        if best_attr:
            most_common_value = candidates[best_attr].mode()[0]
            subset = candidates[candidates[best_attr] == most_common_value]
            guess = subset.sample(1).iloc[0]
            return guess, {"heuristic": "mrv", "attr": best_attr, "value": str(most_common_value), "candidates": len(candidates)}
        
        return candidates.sample(1).iloc[0], {"heuristic": "mrv", "candidates": len(candidates)}
    
    def next_guess_lcv(self, candidates):
        if len(candidates) == 0:
            return None, {}
        if len(candidates) == 1:
            return candidates.iloc[0], {"heuristic": "lcv", "candidates": 1}
        
        best_pokemon = None
        min_avg_elimination = float('inf')
        
        sample_size = min(20, len(candidates))
        sample = candidates.sample(sample_size)
        
        for _, pokemon in sample.iterrows():
            total_elimination = 0
            
            for attr in self.attributes:
                if attr == 'image_url':
                    continue
                value = pokemon[attr]
                if not pd.isna(value):
                    matching = (candidates[attr] == value).sum()
                    elimination = len(candidates) - matching
                    total_elimination += elimination
            
            avg_elimination = total_elimination / len(self.attributes)
            
            if avg_elimination < min_avg_elimination:
                min_avg_elimination = avg_elimination
                best_pokemon = pokemon
        
        return best_pokemon, {"heuristic": "lcv", "avg_elimination": min_avg_elimination, "candidates": len(candidates)}
    
    def next_guess_entropy(self, candidates):
        if len(candidates) == 0:
            return None, {}
        if len(candidates) == 1:
            return candidates.iloc[0], {"heuristic": "entropy", "candidates": 1}
        
        max_entropy = -1
        best_attr = None
        
        for attr in self.attributes:
            if attr == 'image_url':
                continue
            entropy = self.calculate_entropy(candidates, attr)
            if entropy > max_entropy:
                max_entropy = entropy
                best_attr = attr
        
        if best_attr:
            if best_attr in self.numeric_attrs:
                median_value = candidates[best_attr].median()
                distances = (candidates[best_attr] - median_value).abs()
                closest_idx = distances.idxmin()
                guess = candidates.loc[closest_idx]
            else:
                most_common = candidates[best_attr].mode()[0]
                subset = candidates[candidates[best_attr] == most_common]
                guess = subset.sample(1).iloc[0]
            
            return guess, {"heuristic": "entropy", "attr": best_attr, "entropy": max_entropy, "candidates": len(candidates)}
        
        return candidates.sample(1).iloc[0], {"heuristic": "entropy", "candidates": len(candidates)}
    
    def next_guess(self, candidates):
        if self.heuristic == 'mrv':
            return self.next_guess_mrv(candidates)
        elif self.heuristic == 'lcv':
            return self.next_guess_lcv(candidates)
        elif self.heuristic == 'entropy':
            return self.next_guess_entropy(candidates)
        else:
            if len(candidates) == 0:
                return None, {}
            if len(candidates) == 1:
                return candidates.iloc[0], {"heuristic": "random", "candidates": 1}
            return candidates.sample(1).iloc[0], {"heuristic": "random", "candidates": len(candidates)}

# ============ Enhanced GA Solver ============
class GASolver:
    def __init__(self, dataframe, attributes, config: GAConfig):
        self.df = dataframe.copy()
        self.attributes = attributes
        self.config = config
        self.population = self.df.sample(config.pop_size).index.tolist()
        self.feedback_history = []
        self.generation = 0
        self.fitness_cache = {}
        
    def fitness(self, pokemon_idx):
        # Cache fitness calculations
        if pokemon_idx in self.fitness_cache:
            return self.fitness_cache[pokemon_idx]
            
        pokemon = self.df.loc[pokemon_idx]
        score = 0
        penalty_multiplier = 1.0
        
        for guess_idx, feedback in self.feedback_history:
            guess = self.df.loc[guess_idx]
            
            for attr, status in feedback.items():
                if attr == 'image_url':
                    continue
                
                if status == 'green':
                    if pokemon[attr] == guess[attr]:
                        score += 15 * penalty_multiplier
                    else:
                        score -= 30 * penalty_multiplier
                        penalty_multiplier *= 1.2
                
                elif status == 'gray':
                    if attr in ['Type1', 'Type2']:
                        pokemon_types = {pokemon['Type1'], pokemon['Type2']} - {None, float('nan')}
                        pokemon_types = {t for t in pokemon_types if not (isinstance(t, float) and pd.isna(t))}
                        if guess[attr] not in pokemon_types:
                            score += 8
                        else:
                            score -= 15
                    else:
                        if pokemon[attr] != guess[attr]:
                            score += 8
                        else:
                            score -= 15
                
                elif status == 'yellow':
                    pokemon_types = {pokemon['Type1'], pokemon['Type2']} - {None, float('nan')}
                    pokemon_types = {t for t in pokemon_types if not (isinstance(t, float) and pd.isna(t))}
                    if guess[attr] in pokemon_types and pokemon[attr] != guess[attr]:
                        score += 12
                    else:
                        score -= 15
                
                elif status == 'higher':
                    if pokemon[attr] > guess[attr]:
                        score += 12
                    else:
                        score -= 20
                
                elif status == 'lower':
                    if pokemon[attr] < guess[attr]:
                        score += 12
                    else:
                        score -= 20
        
        result = max(0, score)
        self.fitness_cache[pokemon_idx] = result
        return result
    
    def tournament_selection(self):
        tournament = random.sample(self.population, min(self.config.tournament_size, len(self.population)))
        tournament_fitness = [(idx, self.fitness(idx)) for idx in tournament]
        tournament_fitness.sort(key=lambda x: x[1], reverse=True)
        return tournament_fitness[0][0]
    
    def crossover_attribute_blend(self, parent1_idx, parent2_idx):
        """Blend attributes from both parents based on fitness"""
        parent1 = self.df.loc[parent1_idx]
        parent2 = self.df.loc[parent2_idx]
        
        target_attrs = {}
        p1_fitness = self.fitness(parent1_idx)
        p2_fitness = self.fitness(parent2_idx)
        
        for attr in self.attributes:
            if p1_fitness > p2_fitness:
                target_attrs[attr] = parent1[attr] if random.random() < 0.7 else parent2[attr]
            else:
                target_attrs[attr] = parent2[attr] if random.random() < 0.7 else parent1[attr]
        
        return self.find_best_match(target_attrs)
    
    def crossover_uniform(self, parent1_idx, parent2_idx):
        """Uniform crossover - each attribute has 50% chance from either parent"""
        parent1 = self.df.loc[parent1_idx]
        parent2 = self.df.loc[parent2_idx]
        
        target_attrs = {}
        for attr in self.attributes:
            target_attrs[attr] = parent1[attr] if random.random() < 0.5 else parent2[attr]
        
        return self.find_best_match(target_attrs)
    
    def crossover_single_point(self, parent1_idx, parent2_idx):
        """Single-point crossover"""
        parent1 = self.df.loc[parent1_idx]
        parent2 = self.df.loc[parent2_idx]
        
        crossover_point = random.randint(1, len(self.attributes) - 1)
        target_attrs = {}
        
        for i, attr in enumerate(self.attributes):
            target_attrs[attr] = parent1[attr] if i < crossover_point else parent2[attr]
        
        return self.find_best_match(target_attrs)
    
    def crossover_fitness_weighted(self, parent1_idx, parent2_idx):
        """Weighted by fitness - higher fitness parent contributes more"""
        parent1 = self.df.loc[parent1_idx]
        parent2 = self.df.loc[parent2_idx]
        
        p1_fitness = self.fitness(parent1_idx)
        p2_fitness = self.fitness(parent2_idx)
        total_fitness = p1_fitness + p2_fitness + 0.01  # Avoid division by zero
        
        p1_weight = p1_fitness / total_fitness
        
        target_attrs = {}
        for attr in self.attributes:
            target_attrs[attr] = parent1[attr] if random.random() < p1_weight else parent2[attr]
        
        return self.find_best_match(target_attrs)
    
    def find_best_match(self, target_attrs):
        """Find Pokemon that best matches target attributes"""
        best_match_idx = None
        best_match_score = -1
        
        sample_size = min(150, len(self.df))
        candidates = self.df.sample(sample_size)
        
        for idx, row in candidates.iterrows():
            match_score = 0
            for attr in self.attributes:
                if attr == 'image_url':
                    continue
                if not pd.isna(target_attrs[attr]) and not pd.isna(row[attr]):
                    if row[attr] == target_attrs[attr]:
                        match_score += 1
                    elif attr in ['Height', 'Weight']:
                        diff = abs(row[attr] - target_attrs[attr])
                        max_diff = self.df[attr].max() - self.df[attr].min()
                        if max_diff > 0:
                            match_score += 1 - (diff / max_diff)
            
            if match_score > best_match_score:
                best_match_score = match_score
                best_match_idx = idx
        
        return best_match_idx if best_match_idx is not None else candidates.sample(1).index[0]
    
    def crossover(self, parent1_idx, parent2_idx):
        """Perform crossover based on selected strategy"""
        if random.random() > self.config.crossover_rate:
            return parent1_idx if random.random() < 0.5 else parent2_idx
        
        strategy = self.config.crossover_strategy
        
        if strategy == 'uniform':
            return self.crossover_uniform(parent1_idx, parent2_idx)
        elif strategy == 'single_point':
            return self.crossover_single_point(parent1_idx, parent2_idx)
        elif strategy == 'fitness_weighted':
            return self.crossover_fitness_weighted(parent1_idx, parent2_idx)
        else:  # attribute_blend
            return self.crossover_attribute_blend(parent1_idx, parent2_idx)
    
    def mutate(self, pokemon_idx):
        """Adaptive mutation"""
        if random.random() < self.config.mutation_rate:
            fitness = self.fitness(pokemon_idx)
            # Higher mutation chance for low fitness
            if fitness < 10 or random.random() < 0.3:
                return self.df.sample(1).index[0]
        return pokemon_idx
    
    def evolve(self):
        """Run one generation"""
        # Clear fitness cache for new generation
        self.fitness_cache.clear()
        
        # Calculate fitness
        fitness_scores = [(idx, self.fitness(idx)) for idx in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Elite selection
        new_population = [idx for idx, _ in fitness_scores[:self.config.elite_size]]
        
        # Diversity preservation
        diversity_size = max(5, int(self.config.pop_size * 0.1))
        new_population.extend(self.df.sample(diversity_size).index.tolist())
        
        # Generate offspring
        while len(new_population) < self.config.pop_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        self.population = new_population[:self.config.pop_size]
        self.generation += 1
    
    def best_guess(self):
        fitness_scores = [(idx, self.fitness(idx)) for idx in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        return self.df.loc[fitness_scores[0][0]]
    
    def update_feedback(self, guess_idx, feedback):
        self.feedback_history.append((guess_idx, feedback))
        self.fitness_cache.clear()  # Clear cache when new feedback is added
    
    def get_population_stats(self):
        unique_pokemon = len(set(self.population))
        fitness_scores = [self.fitness(idx) for idx in self.population]
        avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0
        max_fitness = max(fitness_scores) if fitness_scores else 0
        min_fitness = min(fitness_scores) if fitness_scores else 0
        
        # Get diversity metrics
        top_10_percent = int(len(fitness_scores) * 0.1)
        top_fitness = sorted(fitness_scores, reverse=True)[:top_10_percent]
        fitness_variance = sum((f - avg_fitness) ** 2 for f in fitness_scores) / len(fitness_scores) if fitness_scores else 0
        
        return {
            "generation": self.generation,
            "unique_pokemon": unique_pokemon,
            "avg_fitness": round(avg_fitness, 2),
            "max_fitness": round(max_fitness, 2),
            "min_fitness": round(min_fitness, 2),
            "fitness_variance": round(fitness_variance, 2),
            "crossover_strategy": self.config.crossover_strategy,
            "population_diversity": round(unique_pokemon / self.config.pop_size * 100, 1)
        }

# ============ API Endpoints ============
@app.get("/")
def root():
    return {"message": "Enhanced Pokedle Solver API", "version": "3.0"}

@app.get("/pokemon")
def get_pokemon_list():
    pokemon_list = []
    for _, row in df.iterrows():
        pokemon_list.append({
            "name": row['Original_Name'],
            "image_url": row.get('image_url', '')
        })
    return {
        "pokemon": pokemon_list,
        "count": len(pokemon_list)
    }

@app.get("/config")
def get_config():
    return {
        "attributes": AVAILABLE_ATTRIBUTES,
        "algorithms": AVAILABLE_ALGORITHMS,
        "heuristics": AVAILABLE_HEURISTICS,
        "heuristic_descriptions": {
            "random": "Random selection from remaining candidates",
            "mrv": "Minimum Remaining Values - choose most constrained attribute",
            "lcv": "Least Constraining Value - minimize future constraint",
            "entropy": "Maximum information gain - highest uncertainty reduction"
        },
        "crossover_strategies": {
            "attribute_blend": "Blend attributes based on parent fitness",
            "uniform": "50-50 chance for each attribute from either parent",
            "single_point": "Single crossover point splits attributes",
            "fitness_weighted": "Higher fitness parent contributes more"
        }
    }

@app.post("/solve")
def solve(config: SolverConfig):
    start_time = time.time()
    
    # Validate
    if config.algorithm not in AVAILABLE_ALGORITHMS:
        raise HTTPException(400, "Invalid algorithm")
    
    if not all(attr in AVAILABLE_ATTRIBUTES for attr in config.attributes):
        raise HTTPException(400, "Invalid attributes")
    
    if config.heuristic not in AVAILABLE_HEURISTICS:
        raise HTTPException(400, "Invalid heuristic")
    
    # Get secret Pokemon
    if config.secret_pokemon:
        secret_match = df[df['Original_Name'] == config.secret_pokemon]
        if secret_match.empty:
            raise HTTPException(400, "Pokemon not found")
        secret = secret_match.iloc[0]
    else:
        secret = df.sample(1).iloc[0]
    
    steps = []
    
    # CSP Algorithm
    if config.algorithm == 'CSP':
        solver = CSPSolver(df, config.attributes, config.heuristic)
        candidates = df
        
        for attempt in range(1, config.max_attempts + 1):
            guess, heuristic_info = solver.next_guess(candidates)
            
            if len(candidates) == 1 or guess is None:
                if guess is not None:
                    steps.append(SolverStep(
                        attempt=attempt,
                        guess_name=guess['Original_Name'],
                        guess_data={attr: str(guess[attr]) for attr in config.attributes},
                        feedback={attr: 'green' for attr in config.attributes},
                        remaining_candidates=1,
                        timestamp=time.time() - start_time,
                        image_url=guess.get('image_url', ''),
                        heuristic_info=heuristic_info
                    ))
                break
            
            feedback = get_feedback(secret, guess, config.attributes)
            
            step = SolverStep(
                attempt=attempt,
                guess_name=guess['Original_Name'],
                guess_data={attr: str(guess[attr]) for attr in config.attributes},
                feedback=feedback,
                remaining_candidates=len(candidates),
                timestamp=time.time() - start_time,
                image_url=guess.get('image_url', ''),
                heuristic_info=heuristic_info
            )
            steps.append(step)
            
            non_image_feedback = {k: v for k, v in feedback.items() if k != 'image_url'}
            if all(v == 'green' for v in non_image_feedback.values()):
                break
            
            for attr, status in feedback.items():
                if attr in ['Height', 'Weight'] and status in ['higher', 'lower']:
                    solver.apply_numeric_feedback(attr, guess[attr], secret[attr])
            
            solver.apply_feedback(guess, feedback)
            candidates = solver.filter_candidates()
    
    # GA Algorithm
    else:
        ga_config = config.ga_config or GAConfig()
        solver = GASolver(df, config.attributes, ga_config)
        
        for attempt in range(1, config.max_attempts + 1):
            # Evolve population
            for _ in range(ga_config.generations_per_guess):
                solver.evolve()
            
            guess = solver.best_guess()
            guess_idx = guess.name
            
            feedback = get_feedback(secret, guess, config.attributes)
            
            pop_stats = solver.get_population_stats()
            
            step = SolverStep(
                attempt=attempt,
                guess_name=guess['Original_Name'],
                guess_data={attr: str(guess[attr]) for attr in config.attributes},
                feedback=feedback,
                remaining_candidates=pop_stats['unique_pokemon'],
                timestamp=time.time() - start_time,
                image_url=guess.get('image_url', ''),
                heuristic_info=pop_stats
            )
            steps.append(step)
            
            non_image_feedback = {k: v for k, v in feedback.items() if k != 'image_url'}
            if all(v == 'green' for v in non_image_feedback.values()):
                break
            
            solver.update_feedback(guess_idx, feedback)
    
    execution_time = time.time() - start_time
    
    return SolverResult(
        secret_name=secret['Original_Name'],
        secret_image=secret.get('image_url', ''),
        success=len(steps) > 0 and all(v == 'green' for v in steps[-1].feedback.values() if v != 'image_url'),
        total_attempts=len(steps),
        steps=steps,
        execution_time=execution_time
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)