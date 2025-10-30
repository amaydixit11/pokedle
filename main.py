# main.py - FastAPI Backend for Pokedle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import random
from typing import List, Optional, Dict
import time

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

# Available attributes and algorithms
AVAILABLE_ATTRIBUTES = ['Generation', 'Height', 'Weight', 'Type1', 'Type2', 'Color', 'evolutionary_stage']
AVAILABLE_ALGORITHMS = ['CSP', 'GA']
AVAILABLE_HEURISTICS = ['random', 'entropy']  # Can add more later

# ============ Models ============
class SolverConfig(BaseModel):
    algorithm: str  # 'CSP' or 'GA'
    attributes: List[str]
    heuristic: str = 'random'
    secret_pokemon: Optional[str] = None  # Name or None for random
    max_attempts: int = 10

class SolverStep(BaseModel):
    attempt: int
    guess_name: str
    guess_data: Dict
    feedback: Dict
    remaining_candidates: int
    timestamp: float

class SolverResult(BaseModel):
    secret_name: str
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

# ============ CSP Solver ============
class PokedleCSP:
    def __init__(self, dataframe, attributes, numeric_attrs=['Height', 'Weight']):
        self.df = dataframe.copy()
        self.attributes = attributes
        self.numeric_attrs = numeric_attrs
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
    
    def next_guess(self, candidates):
        if len(candidates) == 0:
            return None
        if len(candidates) == 1:
            return candidates.iloc[0]
        return candidates.sample(1).iloc[0]

# ============ GA Solver ============
class PokedleGA:
    def __init__(self, dataframe, attributes, pop_size=50, elite_size=10, mutation_rate=0.2):
        self.df = dataframe.copy()
        self.attributes = attributes
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.population = self.df.sample(pop_size).index.tolist()
        self.feedback_history = []
        
    def fitness(self, pokemon_idx):
        pokemon = self.df.loc[pokemon_idx]
        score = 0
        
        for guess_idx, feedback in self.feedback_history:
            guess = self.df.loc[guess_idx]
            
            for attr, status in feedback.items():
                if attr == 'image_url':
                    continue
                
                if status == 'green':
                    if pokemon[attr] == guess[attr]:
                        score += 10
                    else:
                        score -= 20
                elif status == 'gray':
                    if attr in ['Type1', 'Type2']:
                        if guess[attr] not in [pokemon['Type1'], pokemon['Type2']]:
                            score += 5
                        else:
                            score -= 10
                    else:
                        if pokemon[attr] != guess[attr]:
                            score += 5
                        else:
                            score -= 10
                elif status == 'yellow':
                    pokemon_types = {pokemon['Type1'], pokemon['Type2']} - {None, float('nan')}
                    if guess[attr] in pokemon_types and pokemon[attr] != guess[attr]:
                        score += 8
                    else:
                        score -= 10
                elif status == 'higher':
                    if pokemon[attr] > guess[attr]:
                        score += 10
                    else:
                        score -= 15
                elif status == 'lower':
                    if pokemon[attr] < guess[attr]:
                        score += 10
                    else:
                        score -= 15
        
        return max(0, score)
    
    def evolve(self):
        fitness_scores = [(idx, self.fitness(idx)) for idx in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        new_population = [idx for idx, _ in fitness_scores[:self.elite_size]]
        
        while len(new_population) < self.pop_size:
            parent1, parent2 = random.sample([idx for idx, _ in fitness_scores[:20]], 2)
            child = random.choice([parent1, parent2])
            if random.random() < self.mutation_rate:
                child = self.df.sample(1).index[0]
            new_population.append(child)
        
        self.population = new_population
    
    def best_guess(self):
        fitness_scores = [(idx, self.fitness(idx)) for idx in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        return self.df.loc[fitness_scores[0][0]]
    
    def update_feedback(self, guess_idx, feedback):
        self.feedback_history.append((guess_idx, feedback))

# ============ API Endpoints ============
@app.get("/")
def root():
    return {"message": "Pokedle Solver API"}

@app.get("/pokemon")
def get_pokemon_list():
    """Get list of all Pokemon names"""
    return {
        "pokemon": df['Original_Name'].tolist(),
        "count": len(df)
    }

@app.get("/config")
def get_config():
    """Get available configuration options"""
    return {
        "attributes": AVAILABLE_ATTRIBUTES,
        "algorithms": AVAILABLE_ALGORITHMS,
        "heuristics": AVAILABLE_HEURISTICS
    }

@app.post("/solve")
def solve(config: SolverConfig):
    """Run the solver with given configuration"""
    start_time = time.time()
    
    # Validate
    if config.algorithm not in AVAILABLE_ALGORITHMS:
        raise HTTPException(400, "Invalid algorithm")
    
    if not all(attr in AVAILABLE_ATTRIBUTES for attr in config.attributes):
        raise HTTPException(400, "Invalid attributes")
    
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
        solver = PokedleCSP(df, config.attributes)
        candidates = df
        
        for attempt in range(1, config.max_attempts + 1):
            guess = solver.next_guess(candidates)
            
            if len(candidates) == 1 or guess is None:
                if guess is not None:
                    steps.append(SolverStep(
                        attempt=attempt,
                        guess_name=guess['Original_Name'],
                        guess_data={attr: str(guess[attr]) for attr in config.attributes},
                        feedback={attr: 'green' for attr in config.attributes},
                        remaining_candidates=1,
                        timestamp=time.time() - start_time
                    ))
                break
            
            feedback = get_feedback(secret, guess, config.attributes)
            
            step = SolverStep(
                attempt=attempt,
                guess_name=guess['Original_Name'],
                guess_data={attr: str(guess[attr]) for attr in config.attributes},
                feedback=feedback,
                remaining_candidates=len(candidates),
                timestamp=time.time() - start_time
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
    else:  # GA
        solver = PokedleGA(df, config.attributes)
        generations_per_guess = 20
        
        for attempt in range(1, config.max_attempts + 1):
            for _ in range(generations_per_guess):
                solver.evolve()
            
            guess = solver.best_guess()
            guess_idx = guess.name
            
            feedback = get_feedback(secret, guess, config.attributes)
            
            step = SolverStep(
                attempt=attempt,
                guess_name=guess['Original_Name'],
                guess_data={attr: str(guess[attr]) for attr in config.attributes},
                feedback=feedback,
                remaining_candidates=len(set(solver.population)),
                timestamp=time.time() - start_time
            )
            steps.append(step)
            
            non_image_feedback = {k: v for k, v in feedback.items() if k != 'image_url'}
            if all(v == 'green' for v in non_image_feedback.values()):
                break
            
            solver.update_feedback(guess_idx, feedback)
    
    execution_time = time.time() - start_time
    
    return SolverResult(
        secret_name=secret['Original_Name'],
        success=len(steps) > 0 and all(v == 'green' for v in steps[-1].feedback.values()),
        total_attempts=len(steps),
        steps=steps,
        execution_time=execution_time
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)