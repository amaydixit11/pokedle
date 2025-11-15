# Pokedle Solver - Backend API

FastAPI-based REST API implementing four AI algorithms (CSP, GA, A*, SA) for solving the Pokedle game.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [API Endpoints](#api-endpoints)
- [Algorithms](#algorithms)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Development](#development)
- [Testing](#testing)

---

## 🎯 Overview

This backend provides a RESTful API for solving Pokedle puzzles using multiple AI algorithms. Each algorithm is implemented with theoretical correctness and optimized for performance.

### Key Features

- ✅ **4 AI Algorithms**: CSP, GA, A*, SA
- ✅ **Configurable Parameters**: Extensive customization options
- ✅ **Real-time Feedback**: Step-by-step solution tracking
- ✅ **Algorithm Comparison**: Benchmark multiple algorithms
- ✅ **Performance Metrics**: Detailed execution statistics
- ✅ **OpenAPI Documentation**: Auto-generated interactive docs

---

## 🏗️ Architecture
```
backend/
├── main.py                 # FastAPI app & routes
├── config.py               # Global configuration
├── data_loader.py          # Pokemon dataset management
├── feedback.py             # Feedback calculation
├── models.py               # Pydantic data models
├── algorithms/             # Solver implementations
│   ├── base.py            # Abstract base solver
│   ├── csp_solver.py      # CSP with AC-3
│   ├── ga_solver.py       # Genetic algorithm
│   ├── astar_solver.py    # A* search
│   └── simulated_annealing.py  # Simulated annealing
├── heuristics/            # Heuristic functions
│   ├── csp_heuristics.py
│   └── ga_heuristics.py
└── utils/                 # Utility modules
    ├── metrics.py         # Performance calculations
    └── validators.py      # Input validation
```

### Design Patterns

- **Factory Pattern**: Solver creation based on algorithm choice
- **Strategy Pattern**: Interchangeable heuristics
- **Singleton Pattern**: Dataset loader
- **Observer Pattern**: Progress callbacks for streaming

---

## 💻 Installation

### Requirements

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Create Virtual Environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify Dataset**
```bash
# Ensure CSV file exists
ls 03_cleaned_with_images_and_evolutionary_stages.csv
```

4. **Run Server**
```bash
python main.py
# Or with uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

5. **Access API**
- **API Base**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📡 API Endpoints

### Core Endpoints

#### `GET /`
Health check and API info.

**Response:**
```json
{
  "message": "Pokedle Solver API - Logically Correct Version",
  "version": "5.0",
  "improvements": [...]
}
```

---

#### `POST /solve`
Run solver with specified configuration.

**Request Body:**
```json
{
  "algorithm": "CSP",              // Required: CSP, GA, ASTAR, SA
  "attributes": [                  // Required: attribute list
    "Type1", "Type2", "Generation"
  ],
  "secret_pokemon": "Charizard",   // Optional: specific Pokemon or null for random
  "max_attempts": 10,              // Optional: default 10
  "csp_config": {                  // Algorithm-specific config
    "variable_heuristic": "mrv",
    "value_heuristic": "lcv",
    "use_ac3": true
  }
}
```

**CSP Configuration:**
```json
{
  "variable_heuristic": "mrv",     // mrv, degree, mrv_degree, none
  "value_heuristic": "lcv",        // lcv, most_common, none
  "use_ac3": true                  // boolean
}
```

**GA Configuration:**
```json
{
  "pop_size": 50,                  // 10-500
  "elite_size": 10,                // 5-100
  "mutation_rate": 0.2,            // 0.0-1.0
  "crossover_rate": 0.7,           // 0.0-1.0
  "tournament_size": 3,            // 2-20
  "generations_per_guess": 15      // 1-200
}
```

**A* Configuration:**
```json
{
  "beam_width": 100,               // 1+
  "heuristic_weight": 1.0          // 0.0+ (1.0 = admissible)
}
```

**SA Configuration:**
```json
{
  "initial_temp": 100.0,           // > 0
  "cooling_rate": 0.95,            // 0-1
  "min_temp": 0.01,                // > 0
  "iterations_per_temp": 50,       // >= 1
  "reheat_threshold": 0.1          // 0-1
}
```

**Response:**
```json
{
  "secret_name": "Charizard",
  "secret_image": "https://...",
  "success": true,
  "total_attempts": 4,
  "execution_time": 0.856,
  "algorithm": "CSP",
  "algorithm_config": {...},
  "steps": [
    {
      "attempt": 1,
      "guess_name": "Bulbasaur",
      "guess_data": {...},
      "feedback": {...},
      "remaining_candidates": 156,
      "timestamp": 0.123,
      "image_url": "https://...",
      "algorithm_state": {...}
    }
  ],
  "performance_metrics": {
    "avg_time_per_guess": 0.214,
    "efficiency": 0.25,
    "convergence_rate": 0.82
  }
}
```

---

#### `POST /compare`
Compare all algorithms on same Pokemon.

**Request:**
```json
{
  "algorithms": ["CSP", "GA", "ASTAR", "SA"],
  "attributes": ["Type1", "Type2", "Generation"],
  "secret_pokemon": "Pikachu",     // Optional
  "max_attempts": 10
}
```

**Response:**
```json
{
  "secret_pokemon": "Pikachu",
  "winner": "CSP",
  "results": {
    "CSP": {
      "success": true,
      "attempts": 3,
      "time": 0.8,
      "metrics": {...},
      "config": {...}
    },
    "GA": {...},
    "ASTAR": {...},
    "SA": {...}
  },
  "comparison_notes": [...]
}
```

---

#### `GET /config`
Get available configuration options.

**Response:**
```json
{
  "attributes": [
    "Generation", "Height", "Weight", 
    "Type1", "Type2", "Color", "evolutionary_stage"
  ],
  "algorithms": ["CSP", "GA", "ASTAR", "SA"],
  "algorithm_descriptions": {...},
  "csp_heuristics": {
    "variable_ordering": {
      "options": ["mrv", "degree", "mrv_degree", "none"],
      "descriptions": {...}
    },
    "value_ordering": {
      "options": ["lcv", "most_common", "none"],
      "descriptions": {...}
    }
  },
  "default_configs": {...}
}
```

---

#### `GET /pokemon`
Get list of all Pokemon.

**Response:**
```json
{
  "pokemon": [
    {
      "name": "Bulbasaur",
      "image_url": "https://...",
      "generation": 1,
      "type1": "Grass",
      "type2": "Poison"
    }
  ],
  "count": 1010
}
```

---

#### `GET /algorithms/{algorithm}`
Get detailed algorithm information.

**Example:** `GET /algorithms/CSP`

**Response:**
```json
{
  "name": "CSP",
  "description": "Constraint Satisfaction Problem...",
  "config_options": {...},
  "theoretical_properties": [
    "Complete: Guaranteed to find solution",
    "Optimal: With proper heuristics",
    "Systematic: Explores search space systematically"
  ]
}
```

---

#### `GET /algorithm-theory/{algorithm}`
Get theoretical background.

**Response:**
```json
{
  "formulation": {
    "variables": "...",
    "domains": "...",
    "constraints": "..."
  },
  "algorithms": {...},
  "properties": {
    "completeness": "Yes",
    "optimality": "Depends on heuristic",
    "time_complexity": "O(d^n)",
    "space_complexity": "O(n)"
  },
  "correctness": [...]
}
```

---

#### `POST /test/csp-heuristics`
Test all CSP heuristic combinations.

**Request:**
```json
{
  "attributes": ["Type1", "Type2", "Generation"],
  "max_attempts": 10
}
```

**Response:**
```json
{
  "secret_pokemon": "Charizard",
  "results": {
    "mrv+lcv": {"success": true, "attempts": 3},
    "mrv+most_common": {"success": true, "attempts": 4},
    "degree+lcv": {"success": true, "attempts": 5},
    ...
  },
  "best_combination": "mrv+lcv"
}
```

---

## 🧠 Algorithms

### 1. CSP Solver (`algorithms/csp_solver.py`)

**Implementation Details:**
```python
class CSPSolver(BaseSolver):
    def __init__(self, dataframe, attributes, 
                 variable_heuristic='mrv', 
                 value_heuristic='lcv'):
        self.variables = attributes
        self.domains = self._initialize_domains()
        self.constraints = []
        self.assignment = {}
```

**Key Methods:**
- `ac3()`: Arc Consistency Algorithm #3
- `select_unassigned_variable()`: Variable ordering heuristic
- `order_domain_values()`: Value ordering heuristic
- `forward_checking()`: Look-ahead constraint checking
- `add_constraint_from_feedback()`: Convert feedback to constraints

**Heuristics:**

| Variable Ordering | Description | When to Use |
|-------------------|-------------|-------------|
| **MRV** | Minimum Remaining Values | Default, fail-fast |
| **Degree** | Most constrained variable | Complex constraints |
| **MRV+Degree** | MRV with degree tiebreaker | Best of both |
| **None** | No heuristic | Baseline |

| Value Ordering | Description | When to Use |
|----------------|-------------|-------------|
| **LCV** | Least Constraining Value | Default, keeps options open |
| **Most Common** | Most frequent value | Explore likelihood |
| **None** | No heuristic | Baseline |

**Example Usage:**
```python
solver = CSPSolver(
    df=pokemon_df, 
    attributes=['Type1', 'Type2', 'Generation'],
    variable_heuristic='mrv',
    value_heuristic='lcv'
)
guess, state = solver.next_guess()
solver.update_feedback(guess, feedback)
```

---

### 2. GA Solver (`algorithms/ga_solver.py`)

**Implementation Details:**

```python
class GASolver(BaseSolver):
    def __init__(self, dataframe, attributes, config, progress_callback=None):
        self.pop_size = config['pop_size']
        self.mutation_rate = config['mutation_rate']
        self.population = []  # List of Pokemon indices
        self.fitness_cache = {}
```

**Key Methods:**
- `fitness()`: Calculate constraint satisfaction (0-100)
- `tournament_selection()`: Select parents
- `crossover()`: Find valid Pokemon matching parent attributes
- `mutate()`: Replace with similar Pokemon
- `evolve_generation()`: Single evolution step

**Fitness Function:**
```python
def fitness(self, pokemon_idx):
    """
    Fitness = (satisfied_constraints / total_constraints) * 100
    - Hard constraints: 30 points each (must_equal)
    - Soft constraints: 15 points each (not_equal)
    - Numeric constraints: 20 points each (greater/less than)
    """
```

**Crossover Strategy:**
```python
def crossover(self, parent1_idx, parent2_idx):
    """
    1. Blend attributes from both parents (fitness-weighted)
    2. Find VALID Pokemon that best matches target attributes
    3. Never create arbitrary combinations
    """
```

**Example Usage:**
```python
solver = GASolver(
    df=pokemon_df,
    attributes=['Type1', 'Type2', 'Generation'],
    config={
        'pop_size': 50,
        'elite_size': 10,
        'mutation_rate': 0.2,
        'generations_per_guess': 15
    }
)
```

---

### 3. A* Solver (`algorithms/astar_solver.py`)

**Implementation Details:**

```python
class SearchNode:
    def __init__(self, pokemon_idx, g_cost, h_cost, path, parent=None):
        self.pokemon_idx = pokemon_idx
        self.g_cost = g_cost  # Path cost (guesses made)
        self.h_cost = h_cost  # Heuristic (estimated remaining)
        self.f_cost = g_cost + h_cost  # Total cost

class AStarSolver(BaseSolver):
    def __init__(self, dataframe, attributes, config):
        self.open_set = []  # Priority queue (heapq)
        self.closed_set = set()  # Explored Pokemon
        self.beam_width = config['beam_width']
```

**Key Methods:**
- `heuristic()`: Admissible heuristic (never overestimates)
- `get_neighbors()`: Find similar Pokemon
- `is_goal_state()`: Check if solution found
- `rebuild_open_set()`: Update after new feedback

**Admissible Heuristic:**
```python
def heuristic(self, pokemon_idx):
    """
    Count minimum constraint violations.
    Estimate: violations / 2.0 (assume 2 constraints fixed per guess)
    Admissible because it never overestimates.
    """
    min_violations = 0
    for guess_idx, feedback in self.feedback_history:
        for attr, status in feedback.items():
            if violates_constraint(pokemon, attr, status):
                min_violations += penalty
    
    return min_violations / 2.0  # Conservative estimate
```

**Example Usage:**
```python
solver = AStarSolver(
    df=pokemon_df,
    attributes=['Type1', 'Type2', 'Generation'],
    config={
        'beam_width': 100,
        'heuristic_weight': 1.0  # 1.0 = admissible
    }
)
```

---

### 4. SA Solver (`algorithms/simulated_annealing.py`)

**Implementation Details:**

```python
class SimulatedAnnealingSolver(BaseSolver):
    def __init__(self, dataframe, attributes, config):
        self.current_temp = config['initial_temp']
        self.cooling_rate = config['cooling_rate']
        self.current_solution = None
        self.best_solution = None
```

**Key Methods:**
- `energy()`: Count constraint violations (lower = better)
- `acceptance_probability()`: Metropolis criterion
- `get_neighbor()`: Generate similar Pokemon
- `update_constraints_from_feedback()`: Track constraints

**Energy Function:**
```python
def energy(self, pokemon_idx):
    """
    Energy = sum of constraint violations
    - Green mismatch: 10 penalty
    - Gray mismatch: 5 penalty
    - Yellow mismatch: 5 penalty (type not present)
    - Numeric violation: 5 penalty
    Lower energy = better solution
    """
```

**Acceptance:**
```python
def acceptance_probability(self, current_energy, new_energy):
    """
    Always accept if better (new_energy < current_energy)
    Otherwise accept with probability: exp(-ΔE/T)
    """
    if new_energy < current_energy:
        return 1.0
    delta = new_energy - current_energy
    return math.exp(-delta / self.current_temp)
```

**Example Usage:**
```python
solver = SimulatedAnnealingSolver(
    df=pokemon_df,
    attributes=['Type1', 'Type2', 'Generation'],
    config={
        'initial_temp': 100.0,
        'cooling_rate': 0.95,
        'min_temp': 0.01,
        'iterations_per_temp': 50
    }
)
```

---

## ⚙️ Configuration

### Global Configuration (`config.py`)

```python
# Available attributes
AVAILABLE_ATTRIBUTES = [
    'Generation', 'Height', 'Weight', 
    'Type1', 'Type2', 'Color', 'evolutionary_stage'
]

# Numeric attributes (support higher/lower feedback)
NUMERIC_ATTRIBUTES = ['Height', 'Weight']

# Available algorithms
AVAILABLE_ALGORITHMS = ['CSP', 'GA', 'ASTAR', 'SA']

# CSP Heuristics
VARIABLE_ORDERING_HEURISTICS = ['mrv', 'degree', 'mrv_degree', 'none']
VALUE_ORDERING_HEURISTICS = ['lcv', 'most_common', 'none']

# Default configurations
DEFAULT_CSP_CONFIG = {
    'variable_heuristic': 'mrv',
    'value_heuristic': 'lcv',
    'use_ac3': True
}

DEFAULT_GA_CONFIG = {
    'pop_size': 50,
    'elite_size': 10,
    'mutation_rate': 0.2,
    'crossover_rate': 0.7,
    'tournament_size': 3,
    'generations_per_guess': 10
}

DEFAULT_SA_CONFIG = {
    'initial_temp': 100.0,
    'cooling_rate': 0.95,
    'min_temp': 0.01,
    'iterations_per_temp': 50,
    'reheat_threshold': 0.1
}

DEFAULT_ASTAR_CONFIG = {
    'beam_width': 100,
    'heuristic_weight': 1.0
}
```

---

## 📊 Data Format

### Pokemon Dataset CSV

**File**: `03_cleaned_with_images_and_evolutionary_stages.csv`

**Columns:**
- `No`: Pokemon number (1-1010)
- `Original_Name`: Pokemon name (string)
- `Generation`: Generation number (1-9)
- `Height`: Height in meters (float)
- `Weight`: Weight in kg (float)
- `Type1`: Primary type (string)
- `Type2`: Secondary type (string or NaN)
- `Ability1`, `Ability2`, `Ability_Hidden`: Abilities
- `Color`: Pokemon color (string)
- `Egg_Group1`, `Egg_Group2`: Egg groups
- `Category`: Category (string)
- `is_mega`: Mega evolution flag (boolean)
- `image_url`: Official artwork URL (string)
- `evolutionary_stage`: Evolution stage (1-3)

**Example Row:**
```csv
6,Charizard,1,1.7,90.5,Fire,Flying,Blaze,,Solar Power,Red,Monster,Dragon,Ordinary,FALSE,https://raw.githubusercontent.com/.../6.png,3
```

### Feedback Format

Feedback is a dictionary mapping attributes to status:

```python
feedback = {
    'Type1': 'green',      # Exact match
    'Type2': 'yellow',     # Exists but wrong position (types only)
    'Generation': 'gray',  # Does not match
    'Height': 'higher',    # Guess is lower than secret
    'Weight': 'lower'      # Guess is higher than secret
}
```

**Feedback Values:**
- `'green'`: Exact match
- `'yellow'`: Type exists in wrong position (Type1/Type2 only)
- `'gray'`: Does not match / type not present
- `'higher'`: Secret value is higher than guess (numeric)
- `'lower'`: Secret value is lower than guess (numeric)

---

## 🔧 Development

### Project Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Format code
black .
```

### Adding New Algorithm

1. **Create solver file**: `algorithms/new_solver.py`
2. **Inherit from BaseSolver**:
```python
from algorithms.base import BaseSolver

class NewSolver(BaseSolver):
    def next_guess(self):
        # Implementation
        pass
    
    def update_feedback(self, guess, feedback):
        # Implementation
        pass
```

3. **Register in config**: Add to `AVAILABLE_ALGORITHMS`
4. **Update factory**: Add case in `create_solver()` in `main.py`

### Adding New Heuristic

1. **Create heuristic function**: `heuristics/algorithm_heuristics.py`
```python
@staticmethod
def new_heuristic(candidates, attributes, **kwargs):
    # Implementation
    return best_pokemon, info_dict
```

2. **Register in config**: Add to appropriate heuristic list
3. **Add description**: Update `HEURISTIC_DESCRIPTIONS`

### Code Style

- **Docstrings**: Google style
- **Type hints**: Use for all function signatures
- **Comments**: Explain "why", not "what"
- **Naming**: 
  - Functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`

### Manual Testing with cURL

```bash
# Test solve endpoint
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "CSP",
    "attributes": ["Type1", "Type2", "Generation"],
    "secret_pokemon": "Pikachu",
    "max_attempts": 10,
    "csp_config": {
      "variable_heuristic": "mrv",
      "value_heuristic": "lcv",
      "use_ac3": true
    }
  }'

# Test compare endpoint
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "algorithms": ["CSP", "GA", "ASTAR", "SA"],
    "attributes": ["Type1", "Type2"],
    "secret_pokemon": "Charizard",
    "max_attempts": 10
  }'
```

---

## 📈 Performance Optimization

### Tips for Faster Execution

1. **Use CSP for Quick Results**
   - Fastest algorithm with AC-3
   - Optimal with proper heuristics

2. **Reduce GA Parameters**
   - Lower `pop_size` and `generations_per_guess`
   - Balance between speed and quality

3. **Limit Attributes**
   - Fewer attributes = smaller search space
   - Start with 2-3 attributes for testing

4. **Enable Caching**
   - GA fitness cache reduces redundant calculations
   - A* closed set prevents revisiting states

5. **Use Beam Search**
   - A* beam width limits open set size
   - Trade completeness for speed

---

## 📚 API Response Examples

### Successful Solve

```json
{
  "secret_name": "Charizard",
  "secret_image": "https://raw.githubusercontent.com/.../6.png",
  "success": true,
  "total_attempts": 4,
  "execution_time": 0.856,
  "algorithm": "CSP",
  "algorithm_config": {
    "variable_heuristic": "mrv",
    "value_heuristic": "lcv",
    "use_ac3": true
  },
  "steps": [
    {
      "attempt": 1,
      "guess_name": "Bulbasaur",
      "guess_data": {
        "Type1": "Grass",
        "Type2": "Poison",
        "Generation": "1"
      },
      "feedback": {
        "Type1": "gray",
        "Type2": "gray",
        "Generation": "green"
      },
      "remaining_candidates": 156,
      "timestamp": 0.123,
      "image_url": "https://raw.githubusercontent.com/.../1.png",
      "algorithm_state": {
        "algorithm": "CSP",
        "candidates": 156,
        "assignment": {"Generation": "1"},
        "domain_sizes": {
          "Type1": 15,
          "Type2": 12,
          "Generation": 1
        },
        "selected_variable": "Type1",
        "selected_value": "Fire"
      }
    }
    // ... more steps
  ],
  "performance_metrics": {
    "avg_time_per_guess": 0.214,
    "total_guesses": 4,
    "success_rate": 1.0,
    "efficiency": 0.25,
    "convergence_rate": 0.82,
    "diversity_score": 0.75
  }
}
```

### Error Response

```json
{
  "detail": "Invalid algorithm 'INVALID'. Must be one of: CSP, GA, ASTAR, SA"
}
```