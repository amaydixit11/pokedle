import pandas as pd
import heapq
from typing import Dict, Tuple, Any, List, Set, Optional
from algorithms.base import BaseSolver

class SearchNode:
    """
    Node in A* search tree.
    
    CORRECTED: Properly represents search state.
    """
    def __init__(self, pokemon_idx: int, g_cost: float, h_cost: float, 
                 path: List[int], parent=None):
        self.pokemon_idx = pokemon_idx  # Current Pokemon being considered
        self.g_cost = g_cost  # Cost from start (number of guesses so far)
        self.h_cost = h_cost  # Estimated cost to goal (heuristic)
        self.f_cost = g_cost + h_cost  # Total estimated cost
        self.path = path  # Path of guesses that led here
        self.parent = parent  # Parent node
    
    def __lt__(self, other):
        # For heapq: lower f_cost = higher priority
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.pokemon_idx == other.pokemon_idx
    
    def __hash__(self):
        return hash(self.pokemon_idx)

class AStarSolver(BaseSolver):
    """
    A* Search algorithm for Pokedle.
    
    A* formulation:
    
    - State: A Pokemon guess
    - Goal: The secret Pokemon
    - Cost: Number of guesses
    - Heuristic: Estimated remaining guesses based on constraint violations
    
    The challenge: We don't know the goal state (secret Pokemon) initially.
    Solution: Use feedback to narrow search space and guide heuristic.
    """
    
    def __init__(self, dataframe: pd.DataFrame, attributes: list, config: dict):
        super().__init__(dataframe, attributes)
        
        # A* components
        self.open_set = []  # Priority queue of SearchNodes
        self.closed_set = set()  # Pokemon we've already guessed
        self.came_from = {}  # For path reconstruction
        
        # Search space: all Pokemon are initially candidates
        self.candidates = set(dataframe.index)
        
        # Constraints learned from feedback
        self.constraints = {attr: [] for attr in attributes}
        
        # Configuration
        self.beam_width = config.get('beam_width', 100)
        self.heuristic_weight = config.get('heuristic_weight', 1.0)
        
        # Initialize open set with diverse starting Pokemon
        self.initialize_search()
    
    def initialize_search(self):
        """Initialize open set with diverse starting candidates"""
        # Start with a diverse set of Pokemon
        sample_size = min(50, len(self.df))
        initial_candidates = self.df.sample(sample_size).index.tolist()
        
        for idx in initial_candidates:
            h_cost = self.heuristic(idx) * self.heuristic_weight
            node = SearchNode(idx, g_cost=0, h_cost=h_cost, path=[])
            heapq.heappush(self.open_set, node)
    
    def heuristic(self, pokemon_idx: int) -> float:
        """
        Admissible heuristic: estimate minimum guesses to solution.
        
        CORRECTED: Must never overestimate (admissibility requirement).
        
        Strategy:
        - Count MINIMUM constraint violations
        - Each violation requires AT LEAST 1 more guess to fix
        - Use lower bound to ensure admissibility
        
        Returns: Estimated number of guesses remaining (0 = likely the solution)
        """
        if not self.feedback_history:
            # No feedback yet - use diversity heuristic
            return self.diversity_heuristic(pokemon_idx)
        
        pokemon = self.df.loc[pokemon_idx]
        
        # Count minimum violations across all feedback
        min_violations = 0
        satisfied_constraints = 0
        
        for guess_idx, feedback in self.feedback_history:
            guess = self.df.loc[guess_idx]
            
            for attr, status in feedback.items():
                if attr == 'image_url':
                    continue
                
                pokemon_val = pokemon.get(attr)
                guess_val = guess.get(attr)
                
                # Normalize None/NaN
                if pd.isna(pokemon_val):
                    pokemon_val = None
                if pd.isna(guess_val):
                    guess_val = None
                
                if status == 'green':
                    # Must match exactly
                    if pokemon_val == guess_val:
                        satisfied_constraints += 1
                    else:
                        # Clear violation
                        min_violations += 1
                
                elif status == 'gray':
                    # Must not match
                    if attr in ['Type1', 'Type2']:
                        # Check both types
                        type1 = pokemon.get('Type1')
                        type2 = pokemon.get('Type2')
                        if pd.isna(type1):
                            type1 = None
                        if pd.isna(type2):
                            type2 = None
                        
                        if guess_val in [type1, type2]:
                            min_violations += 0.5  # Partial violation
                    else:
                        if pokemon_val == guess_val:
                            min_violations += 1
                
                elif status == 'yellow':
                    # Type exists but wrong position
                    type1 = pokemon.get('Type1')
                    type2 = pokemon.get('Type2')
                    if pd.isna(type1):
                        type1 = None
                    if pd.isna(type2):
                        type2 = None
                    
                    pokemon_types = {type1, type2} - {None}
                    
                    # Must have this type somewhere
                    if guess_val not in pokemon_types:
                        min_violations += 1
                    # But not in current position
                    elif pokemon_val == guess_val:
                        min_violations += 0.5
                
                elif status == 'higher':
                    # Must be greater
                    try:
                        if pokemon_val is not None and guess_val is not None:
                            if float(pokemon_val) <= float(guess_val):
                                # Violation
                                min_violations += 1
                    except (ValueError, TypeError):
                        pass
                
                elif status == 'lower':
                    # Must be less
                    try:
                        if pokemon_val is not None and guess_val is not None:
                            if float(pokemon_val) >= float(guess_val):
                                # Violation
                                min_violations += 1
                    except (ValueError, TypeError):
                        pass
        
        # Admissible heuristic: minimum violations / constraints per guess
        # Assume we can fix at most 2 violations per guess (conservative)
        violations_per_guess = 2.0
        estimated_guesses = min_violations / violations_per_guess
        
        # Add small bonus for constraint satisfaction (encourages progress)
        bonus = -0.1 * satisfied_constraints
        
        return max(0, estimated_guesses + bonus)
    
    def diversity_heuristic(self, pokemon_idx: int) -> float:
        """
        Heuristic based on attribute diversity (used before feedback available).
        
        More common attribute values = higher heuristic (less informative guess).
        """
        pokemon = self.df.loc[pokemon_idx]
        score = 0
        
        for attr in self.attributes:
            if attr == 'image_url':
                continue
            
            value = pokemon.get(attr)
            if pd.isna(value):
                score += 0.5
                continue
            
            # How common is this attribute value?
            frequency = (self.df[attr] == value).sum() / len(self.df)
            score += frequency  # Higher frequency = less informative
        
        return score
    
    def is_goal_state(self, pokemon_idx: int) -> bool:
        """
        Check if this Pokemon satisfies all known constraints.
        
        If true, this is a candidate for the solution.
        """
        return self.heuristic(pokemon_idx) == 0
    
    def get_neighbors(self, current_idx: int) -> List[int]:
        """
        Get neighbor states (Pokemon similar to current).
        
        In this problem, neighbors are Pokemon that share some attributes.
        We limit to candidates that satisfy known constraints.
        """
        if not self.candidates:
            return []
        
        # Return candidates that are similar to current
        current = self.df.loc[current_idx]
        neighbors = []
        
        # Sample candidates for efficiency
        sample_size = min(50, len(self.candidates))
        candidate_sample = list(self.candidates)
        if len(candidate_sample) > sample_size:
            candidate_sample = pd.Series(candidate_sample).sample(sample_size).tolist()
        
        for idx in candidate_sample:
            if idx in self.closed_set:
                continue
            
            neighbor = self.df.loc[idx]
            
            # Calculate similarity
            similarity = 0
            for attr in self.attributes:
                if attr == 'image_url':
                    continue
                
                curr_val = current.get(attr)
                neigh_val = neighbor.get(attr)
                
                if pd.isna(curr_val):
                    curr_val = None
                if pd.isna(neigh_val):
                    neigh_val = None
                
                if curr_val == neigh_val:
                    similarity += 1
            
            # Only include if reasonably similar
            if similarity >= 2:
                neighbors.append(idx)
        
        return neighbors
    
    def update_candidates_from_feedback(self, guess: pd.Series, feedback: Dict[str, str]):
        """
        Update candidate set based on feedback.
        
        Remove Pokemon that violate the feedback constraints.
        """
        valid_candidates = set()
        
        for idx in self.candidates:
            pokemon = self.df.loc[idx]
            valid = True
            
            for attr, status in feedback.items():
                if attr == 'image_url':
                    continue
                
                pokemon_val = pokemon.get(attr)
                guess_val = guess.get(attr)
                
                if pd.isna(pokemon_val):
                    pokemon_val = None
                if pd.isna(guess_val):
                    guess_val = None
                
                if status == 'green':
                    # Must match exactly
                    if pokemon_val != guess_val:
                        valid = False
                        break
                
                elif status == 'gray':
                    if attr in ['Type1', 'Type2']:
                        # Type must not appear anywhere
                        type1 = pokemon.get('Type1')
                        type2 = pokemon.get('Type2')
                        if pd.isna(type1):
                            type1 = None
                        if pd.isna(type2):
                            type2 = None
                        
                        if guess_val in [type1, type2]:
                            valid = False
                            break
                    else:
                        if pokemon_val == guess_val:
                            valid = False
                            break
                
                elif status == 'yellow':
                    # Type exists but wrong position
                    type1 = pokemon.get('Type1')
                    type2 = pokemon.get('Type2')
                    if pd.isna(type1):
                        type1 = None
                    if pd.isna(type2):
                        type2 = None
                    
                    pokemon_types = {type1, type2} - {None}
                    
                    # Must have this type somewhere
                    if guess_val not in pokemon_types:
                        valid = False
                        break
                    
                    # But not in this position
                    if pokemon_val == guess_val:
                        valid = False
                        break
                
                elif status == 'higher':
                    try:
                        if pokemon_val is not None and guess_val is not None:
                            if float(pokemon_val) <= float(guess_val):
                                valid = False
                                break
                    except (ValueError, TypeError):
                        pass
                
                elif status == 'lower':
                    try:
                        if pokemon_val is not None and guess_val is not None:
                            if float(pokemon_val) >= float(guess_val):
                                valid = False
                                break
                    except (ValueError, TypeError):
                        pass
            
            if valid:
                valid_candidates.add(idx)
        
        self.candidates = valid_candidates
    
    def next_guess(self) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Generate next guess using A* search.
        
        Strategy:
        1. Pop node with lowest f_cost from open set
        2. If goal state, return it
        3. Otherwise, expand neighbors and add to open set
        4. Use beam search to limit open set size
        """
        if not self.open_set:
            # Fallback: return best candidate
            if self.candidates:
                best_idx = min(self.candidates, key=self.heuristic)
                pokemon = self.df.loc[best_idx]
                return pokemon, {"algorithm": "astar", "fallback": True}
            return None, {"error": "no_candidates"}
        
        # Beam search: keep only best nodes
        if len(self.open_set) > self.beam_width:
            self.open_set = heapq.nsmallest(self.beam_width, self.open_set)
            heapq.heapify(self.open_set)
        
        # CAPTURE COMPLETE OPEN SET STATE BEFORE POPPING
        open_set_snapshot = []
        for node in list(self.open_set):
            try:
                pokemon_name = self.df.loc[node.pokemon_idx]['Original_Name']
                open_set_snapshot.append({
                    "pokemon_idx": int(node.pokemon_idx),
                    "pokemon_name": str(pokemon_name),
                    "g_cost": float(node.g_cost),
                    "h_cost": round(float(node.h_cost), 3),
                    "f_cost": round(float(node.f_cost), 3),
                    "parent_idx": int(node.parent.pokemon_idx) if node.parent else None,
                    "path": [int(p) for p in node.path] if node.path else []
                })
            except Exception as e:
                print(f"Error capturing node {node.pokemon_idx}: {e}")
                continue
        
        # Pop node with lowest f_cost
        current_node = heapq.heappop(self.open_set)
        current_idx = current_node.pokemon_idx
        
        # Add to closed set (already guessed)
        self.closed_set.add(current_idx)
        
        # Capture closed set for visualization
        closed_set_snapshot = []
        for idx in self.closed_set:
            try:
                closed_set_snapshot.append({
                    "pokemon_idx": int(idx),
                    "pokemon_name": str(self.df.loc[idx]['Original_Name'])
                })
            except:
                continue
        
        # Check if goal state
        if self.is_goal_state(current_idx):
            pokemon = self.df.loc[current_idx]
            info = {
                "algorithm": "astar",
                "g_cost": current_node.g_cost,
                "h_cost": round(current_node.h_cost, 3),
                "f_cost": round(current_node.f_cost, 3),
                "path_length": len(current_node.path),
                "candidates": len(self.candidates),
                "goal_state": True,
                "open_set_nodes": open_set_snapshot
            }
            return pokemon, info
        
        # Expand neighbors (for future iterations)
        neighbors = self.get_neighbors(current_idx)
        
        for neighbor_idx in neighbors:
            if neighbor_idx in self.closed_set:
                continue
            
            # Cost from start to neighbor
            g_cost = current_node.g_cost + 1
            
            # Estimated cost from neighbor to goal
            h_cost = self.heuristic(neighbor_idx) * self.heuristic_weight
            
            # Create neighbor node
            neighbor_node = SearchNode(
                neighbor_idx,
                g_cost,
                h_cost,
                current_node.path + [current_idx],
                parent=current_node
            )
            
            heapq.heappush(self.open_set, neighbor_node)
        
        # Return current node as guess
        pokemon = self.df.loc[current_idx]
        
        info = {
            "algorithm": "astar",
            "g_cost": current_node.g_cost,
            "h_cost": round(current_node.h_cost, 3),
            "f_cost": round(current_node.f_cost, 3),
            "open_set_size": len(self.open_set),
            "closed_set_size": len(self.closed_set),
            "candidates": len(self.candidates),
            "open_set_nodes": open_set_snapshot,  # ALL open set nodes!
            "closed_set_nodes": closed_set_snapshot,  # ALL closed set nodes!
            "current_node": {
                "pokemon_idx": int(current_idx),
                "pokemon_name": str(self.df.loc[current_idx]['Original_Name']),
                "path": [int(p) for p in current_node.path] if current_node.path else []
            }
        }
        
        return pokemon, info
    
    def update_feedback(self, guess: pd.Series, feedback: Dict[str, str]):
        """Update search state with new feedback"""
        guess_idx = guess.name
        self.add_feedback(guess_idx, feedback)
        
        # Update candidates based on feedback
        self.update_candidates_from_feedback(guess, feedback)
        
        # Rebuild open set with updated heuristics
        self.rebuild_open_set()
    
    def rebuild_open_set(self):
        """
        Rebuild open set with updated heuristics.
        
        After new feedback, heuristic values change, so we need to update priorities.
        """
        # Extract all nodes from open set
        nodes = []
        while self.open_set:
            nodes.append(heapq.heappop(self.open_set))
        
        # Re-add nodes that are still candidates with updated heuristics
        for node in nodes:
            if node.pokemon_idx in self.candidates and node.pokemon_idx not in self.closed_set:
                # Recalculate heuristic
                h_cost = self.heuristic(node.pokemon_idx) * self.heuristic_weight
                new_node = SearchNode(
                    node.pokemon_idx,
                    node.g_cost,
                    h_cost,
                    node.path,
                    node.parent
                )
                heapq.heappush(self.open_set, new_node)
        
        # Add new candidates to open set
        for idx in self.candidates:
            if idx not in self.closed_set:
                # Check if already in open set
                if not any(node.pokemon_idx == idx for node in self.open_set):
                    g_cost = len(self.feedback_history)
                    h_cost = self.heuristic(idx) * self.heuristic_weight
                    node = SearchNode(idx, g_cost, h_cost, [])
                    heapq.heappush(self.open_set, node)
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information"""
        return {
            "algorithm": "A*",
            "open_set_size": len(self.open_set),
            "closed_set_size": len(self.closed_set),
            "candidates": len(self.candidates),
            "heuristic_weight": self.heuristic_weight,
            "beam_width": self.beam_width
        }