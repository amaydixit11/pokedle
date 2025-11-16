import pandas as pd
import math
from typing import Tuple, Dict, Any

class CSPHeuristics:
    """Collection of heuristics for CSP solving"""
    
    @staticmethod
    def random(candidates: pd.DataFrame, attributes: list) -> Tuple[pd.Series, Dict]:
        """Random selection"""
        if len(candidates) == 0:
            return None, {}
        return candidates.sample(1).iloc[0], {
            "heuristic": "random",
            "candidates": len(candidates)
        }
    
    @staticmethod
    def mrv(candidates: pd.DataFrame, attributes: list) -> Tuple[pd.Series, Dict]:
        """
        Minimum Remaining Values:
        Choose variable with fewest remaining values in domain.
        """
        if len(candidates) == 0:
            return None, {}
        if len(candidates) == 1:
            return candidates.iloc[0], {"heuristic": "mrv", "candidates": 1}
        
        min_values = float('inf')
        best_attr = None
        
        for attr in attributes:
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
            return guess, {
                "heuristic": "mrv",
                "attr": best_attr,
                "unique_values": min_values,
                "value": str(most_common_value),
                "candidates": len(candidates)
            }
        
        return candidates.sample(1).iloc[0], {"heuristic": "mrv", "candidates": len(candidates)}
    
    @staticmethod
    def lcv(candidates: pd.DataFrame, attributes: list) -> Tuple[pd.Series, Dict]:
        """
        Least Constraining Value:
        Choose value that rules out fewest values for remaining variables.
        """
        if len(candidates) == 0:
            return None, {}
        if len(candidates) == 1:
            return candidates.iloc[0], {"heuristic": "lcv", "candidates": 1}
        
        best_pokemon = None
        min_avg_elimination = float('inf')
        
        sample_size = min(30, len(candidates))
        sample = candidates.sample(sample_size)
        
        for _, pokemon in sample.iterrows():
            total_elimination = 0
            
            for attr in attributes:
                if attr == 'image_url':
                    continue
                value = pokemon[attr]
                if not pd.isna(value):
                    matching = (candidates[attr] == value).sum()
                    elimination = len(candidates) - matching
                    total_elimination += elimination
            
            avg_elimination = total_elimination / len(attributes)
            
            if avg_elimination < min_avg_elimination:
                min_avg_elimination = avg_elimination
                best_pokemon = pokemon
        
        return best_pokemon, {
            "heuristic": "lcv",
            "avg_elimination": round(min_avg_elimination, 2),
            "candidates": len(candidates)
        }
    
    @staticmethod
    def entropy(candidates: pd.DataFrame, attributes: list) -> Tuple[pd.Series, Dict]:
        """
        Maximum Entropy:
        Choose attribute with highest information entropy.
        """
        if len(candidates) == 0:
            return None, {}
        if len(candidates) == 1:
            return candidates.iloc[0], {"heuristic": "entropy", "candidates": 1}
        
        max_entropy = -1
        best_attr = None
        
        for attr in attributes:
            if attr == 'image_url':
                continue
            
            value_counts = candidates[attr].value_counts()
            total = len(candidates)
            entropy = 0
            
            for count in value_counts:
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)
            
            if entropy > max_entropy:
                max_entropy = entropy
                best_attr = attr
        
        if best_attr:
            numeric_attrs = ['Height', 'Weight']
            if best_attr in numeric_attrs:
                median_value = candidates[best_attr].median()
                distances = (candidates[best_attr] - median_value).abs()
                closest_idx = distances.idxmin()
                guess = candidates.loc[closest_idx]
            else:
                most_common = candidates[best_attr].mode()[0]
                subset = candidates[candidates[best_attr] == most_common]
                guess = subset.sample(1).iloc[0]
            
            return guess, {
                "heuristic": "entropy",
                "attr": best_attr,
                "entropy": round(max_entropy, 3),
                "candidates": len(candidates)
            }
        
        return candidates.sample(1).iloc[0], {"heuristic": "entropy", "candidates": len(candidates)}
    
    @staticmethod
    def degree(candidates: pd.DataFrame, attributes: list, constraints: dict) -> Tuple[pd.Series, Dict]:
        """
        Degree Heuristic:
        Choose variable involved in most constraints.
        """
        if len(candidates) == 0:
            return None, {}
        if len(candidates) == 1:
            return candidates.iloc[0], {"heuristic": "degree", "candidates": 1}
        
        # Count constraints per attribute
        constraint_counts = {attr: len(cons) for attr, cons in constraints.items()}
        
        # Find attribute with most constraints
        if constraint_counts:
            best_attr = max(constraint_counts, key=constraint_counts.get)
            max_constraints = constraint_counts[best_attr]
            
            if max_constraints > 0:
                # Select pokemon that best satisfies this attribute
                most_common = candidates[best_attr].mode()[0]
                subset = candidates[candidates[best_attr] == most_common]
                guess = subset.sample(1).iloc[0]
                
                return guess, {
                    "heuristic": "degree",
                    "attr": best_attr,
                    "constraints": max_constraints,
                    "candidates": len(candidates)
                }
        
        return candidates.sample(1).iloc[0], {"heuristic": "degree", "candidates": len(candidates)}
    
    @staticmethod
    def forward_checking(candidates: pd.DataFrame, attributes: list, constraints: dict) -> Tuple[pd.Series, Dict]:
        """
        Forward Checking:
        Look ahead to see which choice leaves most options.
        """
        if len(candidates) == 0:
            return None, {}
        if len(candidates) == 1:
            return candidates.iloc[0], {"heuristic": "forward_checking", "candidates": 1}
        
        best_pokemon = None
        max_remaining = -1
        
        sample_size = min(20, len(candidates))
        sample = candidates.sample(sample_size)
        
        for _, pokemon in sample.iterrows():
            # Simulate choosing this pokemon
            # Count how many candidates would remain feasible
            remaining = 0
            
            for _, candidate in candidates.iterrows():
                feasible = True
                
                for attr in attributes:
                    if attr == 'image_url':
                        continue
                    
                    # Check if candidate could still be valid
                    if pokemon[attr] != candidate[attr]:
                        # Would create new constraint
                        # Check if candidate satisfies existing constraints
                        for op, val in constraints.get(attr, []):
                            if op == '==' and candidate[attr] != val:
                                feasible = False
                                break
                            elif op == '!=' and candidate[attr] == val:
                                feasible = False
                                break
                    
                    if not feasible:
                        break
                
                if feasible:
                    remaining += 1
            
            if remaining > max_remaining:
                max_remaining = remaining
                best_pokemon = pokemon
        
        if best_pokemon is not None:
            return best_pokemon, {
                "heuristic": "forward_checking",
                "remaining_after": max_remaining,
                "candidates": len(candidates)
            }
        
        return candidates.sample(1).iloc[0], {"heuristic": "forward_checking", "candidates": len(candidates)}
    
    @staticmethod
    def domain_wipeout(candidates: pd.DataFrame, attributes: list) -> Tuple[pd.Series, Dict]:
        """
        Domain Wipeout Prevention:
        Avoid choices that would eliminate all remaining candidates.
        """
        if len(candidates) == 0:
            return None, {}
        if len(candidates) == 1:
            return candidates.iloc[0], {"heuristic": "domain_wipeout", "candidates": 1}
        
        # Find pokemon that preserves maximum diversity
        best_pokemon = None
        max_diversity = -1
        
        sample_size = min(25, len(candidates))
        sample = candidates.sample(sample_size)
        
        for _, pokemon in sample.iterrows():
            diversity_score = 0
            
            for attr in attributes:
                if attr == 'image_url':
                    continue
                
                value = pokemon[attr]
                if pd.isna(value):
                    continue
                
                # Count unique values that would remain
                matching = candidates[candidates[attr] == value]
                non_matching = candidates[candidates[attr] != value]
                
                # Prefer choices that keep both options open
                diversity_score += min(len(matching), len(non_matching))
            
            if diversity_score > max_diversity:
                max_diversity = diversity_score
                best_pokemon = pokemon
        
        if best_pokemon is not None:
            return best_pokemon, {
                "heuristic": "domain_wipeout",
                "diversity_score": max_diversity,
                "candidates": len(candidates)
            }
        
