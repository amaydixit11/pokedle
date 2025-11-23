import pandas as pd
import math
from typing import Dict, Tuple, Any, List, Set, Optional
from collections import deque
from algorithms.base import BaseSolver

class CSPSolver(BaseSolver):
    """
    Constraint Satisfaction Problem solver for Pokedle.
    
    CORRECTED VERSION with proper CSP formulation:
    - Variables: Attributes to guess (Type1, Type2, Height, etc.)
    - Domains: Possible values for each variable
    - Constraints: Rules derived from feedback
    
    Uses two types of heuristics:
    1. Variable ordering: Which attribute to constrain next (MRV, Degree, etc.)
    2. Value ordering: Which value to try for that attribute (LCV, etc.)
    """
    
    def __init__(self, dataframe: pd.DataFrame, attributes: list, 
                 variable_heuristic: str = 'mrv', 
                 value_heuristic: str = 'lcv'):
        super().__init__(dataframe, attributes)
        
        # CSP components
        self.variables = [attr for attr in attributes if attr != 'image_url']
        self.domains = self._initialize_domains()
        self.constraints = []  # List of constraint functions
        self.arcs = []  # For AC-3
        
        # Heuristics
        self.variable_heuristic = variable_heuristic  # Which variable to assign next
        self.value_heuristic = value_heuristic  # Which value to try first
        
        # Current partial assignment
        self.assignment = {}
        
        # Candidate Pokemon (those consistent with assignment)
        self.candidates = set(dataframe.index)
        
    def _initialize_domains(self) -> Dict[str, Set]:
        """Initialize domain for each variable (attribute)"""
        domains = {}
        
        for var in self.variables:
            # Domain is all possible values for this attribute
            unique_values = self.df[var].dropna().unique()
            domains[var] = set(unique_values)
            
            # For Type2, also include None (Pokemon can have no second type)
            if var == 'Type2':
                if self.df[var].isna().any():
                    domains[var].add(None)
        
        return domains
    
    def add_constraint_from_feedback(self, guess: pd.Series, feedback: Dict[str, str]):
        """
        Convert feedback into CSP constraints.
        
        This is the key correctness improvement: properly modeling feedback as constraints.
        """
        guess_values = {}
        for var in self.variables:
            val = guess.get(var)
            if pd.isna(val):
                val = None
            guess_values[var] = val
        
        for var, status in feedback.items():
            if var not in self.variables:
                continue
            
            guess_val = guess_values[var]
            
            if status == 'green':
                # Unary constraint: variable MUST equal this value
                self.add_unary_constraint(var, lambda v, target=guess_val: v == target)
                # Update assignment
                self.assignment[var] = guess_val
                # Reduce domain to single value
                self.domains[var] = {guess_val}
                
            elif status == 'gray':
                if var in ['Type1', 'Type2']:
                    # Type doesn't exist anywhere in the Pokemon
                    # Binary constraint: both Type1 and Type2 must not be this
                    self.add_unary_constraint(var, lambda v, target=guess_val: v != target)
                    # Also constrain the other type variable
                    other_type = 'Type2' if var == 'Type1' else 'Type1'
                    self.add_unary_constraint(other_type, lambda v, target=guess_val: v != target)
                    
                    # Remove from domains
                    self.domains[var].discard(guess_val)
                    if other_type in self.domains:
                        self.domains[other_type].discard(guess_val)
                else:
                    # Not equal constraint
                    self.add_unary_constraint(var, lambda v, target=guess_val: v != target)
                    self.domains[var].discard(guess_val)
                    
            elif status == 'yellow':
                # Type exists but in wrong position (only for Type1/Type2)
                if var in ['Type1', 'Type2']:
                    # This variable cannot be this value
                    self.add_unary_constraint(var, lambda v, target=guess_val: v != target)
                    self.domains[var].discard(guess_val)
                    
                    # But the OTHER type variable MUST be this value
                    other_type = 'Type2' if var == 'Type1' else 'Type1'
                    self.add_unary_constraint(other_type, lambda v, target=guess_val: v == target)
                    self.assignment[other_type] = guess_val
                    self.domains[other_type] = {guess_val}
                    
            elif status == 'higher':
                # Numeric constraint: value must be greater than guess
                self.add_unary_constraint(var, lambda v, target=guess_val: 
                                        v is not None and float(v) > float(target))
                # Update domain
                self.domains[var] = {v for v in self.domains[var] 
                                    if v is not None and float(v) > float(guess_val)}
                                    
            elif status == 'lower':
                # Numeric constraint: value must be less than guess
                self.add_unary_constraint(var, lambda v, target=guess_val: 
                                        v is not None and float(v) < float(target))
                # Update domain
                self.domains[var] = {v for v in self.domains[var] 
                                    if v is not None and float(v) < float(guess_val)}
    
    def add_unary_constraint(self, variable: str, predicate):
        """Add a unary constraint on a variable"""
        def constraint(assignment):
            if variable not in assignment:
                return True
            return predicate(assignment[variable])
        
        self.constraints.append((constraint, [variable]))
    
    def add_binary_constraint(self, var1: str, var2: str, predicate):
        """Add a binary constraint between two variables"""
        def constraint(assignment):
            if var1 not in assignment or var2 not in assignment:
                return True
            return predicate(assignment[var1], assignment[var2])
        
        self.constraints.append((constraint, [var1, var2]))
        # Add arcs for AC-3
        self.arcs.append((var1, var2))
        self.arcs.append((var2, var1))
    
    def is_consistent(self, assignment: Dict) -> bool:
        """Check if assignment satisfies all constraints"""
        for constraint, variables in self.constraints:
            # Check if all variables in constraint are assigned
            if all(var in assignment for var in variables):
                if not constraint(assignment):
                    return False
        return True
    
    def ac3(self) -> bool:
        """
        AC-3 algorithm for arc consistency.
        
        Returns True if domains are consistent, False if inconsistency detected.
        """
        queue = deque(self.arcs)
        
        while queue:
            (xi, xj) = queue.popleft()
            
            if self.revise(xi, xj):
                if len(self.domains[xi]) == 0:
                    return False  # Domain wipeout
                
                # Add all arcs (xk, xi) where xk is a neighbor of xi
                for xk in self.get_neighbors(xi):
                    if xk != xj:
                        queue.append((xk, xi))
        
        return True
    
    def revise(self, xi: str, xj: str) -> bool:
        """
        Revise domain of xi based on constraints with xj.
        
        Returns True if domain of xi was revised.
        """
        revised = False
        
        # Find constraints involving xi and xj
        for constraint, variables in self.constraints:
            if set(variables) == {xi, xj}:
                # For each value in xi's domain
                for value_i in list(self.domains[xi]):
                    # Check if there exists a value in xj's domain that satisfies constraint
                    satisfiable = False
                    
                    for value_j in self.domains[xj]:
                        test_assignment = {xi: value_i, xj: value_j}
                        if constraint(test_assignment):
                            satisfiable = True
                            break
                    
                    # If no value in xj's domain satisfies constraint, remove value_i
                    if not satisfiable:
                        self.domains[xi].discard(value_i)
                        revised = True
        
        return revised
    
    def get_neighbors(self, variable: str) -> List[str]:
        """Get all variables that share a constraint with given variable"""
        neighbors = set()
        for _, variables in self.constraints:
            if variable in variables:
                neighbors.update(v for v in variables if v != variable)
        return list(neighbors)
    
    def select_unassigned_variable(self) -> Optional[str]:
        """
        VARIABLE ORDERING HEURISTIC
        
        Select which variable (attribute) to assign next.
        """
        unassigned = [v for v in self.variables if v not in self.assignment]
        
        if not unassigned:
            return None
        
        if self.variable_heuristic == 'mrv':
            # Minimum Remaining Values: choose variable with smallest domain
            return min(unassigned, key=lambda v: len(self.domains[v]))
        
        elif self.variable_heuristic == 'degree':
            # Degree heuristic: choose variable with most constraints
            return max(unassigned, key=lambda v: len(self.get_neighbors(v)))
        
        elif self.variable_heuristic == 'mrv_degree':
            # MRV with degree as tiebreaker
            min_domain_size = min(len(self.domains[v]) for v in unassigned)
            candidates = [v for v in unassigned if len(self.domains[v]) == min_domain_size]
            return max(candidates, key=lambda v: len(self.get_neighbors(v)))
        
        else:  # 'none' or unknown
            return unassigned[0]
    
    def order_domain_values(self, variable: str) -> List:
        """
        VALUE ORDERING HEURISTIC
        
        Order the values in the domain of a variable.
        """
        domain = list(self.domains[variable])
        
        if self.value_heuristic == 'lcv':
            # Least Constraining Value: prefer values that rule out fewest values in neighbors
            def count_constraints(value):
                count = 0
                test_assignment = dict(self.assignment)
                test_assignment[variable] = value
                
                # Count how many values this rules out in neighboring variables
                for neighbor in self.get_neighbors(variable):
                    if neighbor in self.assignment:
                        continue
                    
                    for neighbor_value in self.domains[neighbor]:
                        test_assignment[neighbor] = neighbor_value
                        if not self.is_consistent(test_assignment):
                            count += 1
                
                return count
            
            return sorted(domain, key=count_constraints)
        
        elif self.value_heuristic == 'most_common':
            # Choose values that appear most frequently in remaining candidates
            def frequency(value):
                return sum(1 for idx in self.candidates 
                          if self.df.loc[idx, variable] == value)
            
            return sorted(domain, key=frequency, reverse=True)
        
        else:  # 'none' or unknown
            return domain
    
    def forward_checking(self, variable: str, value: Any) -> Dict[str, Set]:
        """
        Perform forward checking after assigning variable=value.
        
        Returns the pruned domains, or None if inconsistency detected.
        """
        pruned = {}
        
        # For each unassigned neighbor
        for neighbor in self.get_neighbors(variable):
            if neighbor in self.assignment:
                continue
            
            pruned[neighbor] = set()
            
            # Check each value in neighbor's domain
            for neighbor_value in list(self.domains[neighbor]):
                test_assignment = dict(self.assignment)
                test_assignment[variable] = value
                test_assignment[neighbor] = neighbor_value
                
                if not self.is_consistent(test_assignment):
                    pruned[neighbor].add(neighbor_value)
            
            # If all values pruned, inconsistency
            if len(pruned[neighbor]) == len(self.domains[neighbor]):
                return None
        
        return pruned
    
    def next_guess(self) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Generate next guess using CSP solving.
        
        Strategy:
        1. Use variable ordering heuristic to select next attribute to constrain
        2. Use value ordering heuristic to select best value for that attribute
        3. Apply AC-3 for constraint propagation
        4. Find Pokemon that matches current partial assignment
        """
        # Apply AC-3 for constraint propagation
        if not self.ac3():
            # Inconsistency detected - should not happen with correct implementation
            return None, {"error": "domain_wipeout", "algorithm": "CSP"}
        
        # Update candidates based on current domains and assignment
        self.update_candidates()
        
        if len(self.candidates) == 0:
            return None, {"error": "no_candidates", "algorithm": "CSP"}
        
        if len(self.candidates) == 1:
            # Only one candidate left - return it
            pokemon = self.df.loc[list(self.candidates)[0]]
            return pokemon, {
                "algorithm": "CSP",
                "variable_heuristic": self.variable_heuristic,
                "value_heuristic": self.value_heuristic,
                "candidates": 1,
                "domains": {v: len(d) for v, d in self.domains.items()}
            }
        
        # Select variable to assign using variable ordering heuristic
        variable = self.select_unassigned_variable()
        
        if variable is None:
            # All variables assigned - pick any consistent Pokemon
            pokemon = self.df.loc[list(self.candidates)[0]]
            return pokemon, {
                "algorithm": "CSP",
                "candidates": len(self.candidates),
                "fully_assigned": True
            }
        
        # Order values using value ordering heuristic
        ordered_values = self.order_domain_values(variable)
        
        if not ordered_values:
            # No values in domain - should not happen after AC-3
            pokemon = self.df.loc[list(self.candidates)[0]]
            return pokemon, {"algorithm": "CSP", "fallback": True}
        
        # Choose the best value according to heuristic
        best_value = ordered_values[0]
        
        # Find a Pokemon that matches current assignment + new value
        test_assignment = dict(self.assignment)
        test_assignment[variable] = best_value
        
        matching_pokemon = self.find_matching_pokemon(test_assignment)
        
        if matching_pokemon is None:
            # Fallback: return any candidate
            pokemon = self.df.loc[list(self.candidates)[0]]
            return pokemon, {"algorithm": "CSP", "fallback": True}
        
        info = {
            "algorithm": "CSP",
            "variable_heuristic": self.variable_heuristic,
            "value_heuristic": self.value_heuristic,
            "selected_variable": variable,
            "selected_value": str(best_value),
            "candidates": len(self.candidates),
            "domains": {v: len(d) for v, d in self.domains.items()},
            "assignment_size": len(self.assignment)
        }
        
        return matching_pokemon, info
    
    def find_matching_pokemon(self, assignment: Dict) -> Optional[pd.Series]:
        """Find a Pokemon that matches the partial assignment"""
        for idx in self.candidates:
            pokemon = self.df.loc[idx]
            matches = True
            
            for var, value in assignment.items():
                pokemon_val = pokemon[var]
                if pd.isna(pokemon_val):
                    pokemon_val = None
                
                if pokemon_val != value:
                    matches = False
                    break
            
            if matches:
                return pokemon
        
        return None
    
    def update_candidates(self):
        """Update candidate set based on current domains and assignment"""
        valid_candidates = set()
        
        for idx in range(len(self.df)):
            pokemon = self.df.iloc[idx]
            valid = True
            
            # Check if Pokemon is consistent with assignment
            for var, value in self.assignment.items():
                pokemon_val = pokemon[var]
                if pd.isna(pokemon_val):
                    pokemon_val = None
                
                if pokemon_val != value:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if Pokemon values are in domains
            for var in self.variables:
                if var in self.assignment:
                    continue  # Already checked
                
                pokemon_val = pokemon[var]
                if pd.isna(pokemon_val):
                    pokemon_val = None
                
                if pokemon_val not in self.domains[var]:
                    valid = False
                    break
            
            if valid:
                valid_candidates.add(idx)
        
        self.candidates = valid_candidates
    
    def update_feedback(self, guess: pd.Series, feedback: Dict[str, str]):
        """Update solver with new feedback"""
        guess_idx = guess.name
        self.add_feedback(guess_idx, feedback)
        
        # Convert feedback to constraints
        self.add_constraint_from_feedback(guess, feedback)
        
        # Apply AC-3
        self.ac3()
        
        # Update candidates
        self.update_candidates()
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information"""
        return {
            "algorithm": "CSP",
            "variable_heuristic": self.variable_heuristic,
            "value_heuristic": self.value_heuristic,
            "candidates": len(self.candidates),
            "assignment": dict(self.assignment),
            "domains": {v: list(d)[:5] for v, d in self.domains.items()},  # Show first 5
            "domain_sizes": {v: len(d) for v, d in self.domains.items()},
            "num_constraints": len(self.constraints),
            "variables_assigned": len(self.assignment),
            "variables_remaining": len(self.variables) - len(self.assignment)
        }
    