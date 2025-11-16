from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Dict, Any

class BaseHeuristic(ABC):
    """Abstract base class for heuristics"""
    
    @abstractmethod
    def select(self, candidates: pd.DataFrame, attributes: list, **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Select next candidate based on heuristic.
        
        Args:
            candidates: DataFrame of remaining candidates
            attributes: List of attributes to consider
            **kwargs: Additional parameters specific to heuristic
            
        Returns:
            Tuple of (selected_pokemon, info_dict)
        """
        pass
    
    def validate_candidates(self, candidates: pd.DataFrame) -> bool:
        """Validate that candidates DataFrame is not empty"""
        return candidates is not None and len(candidates) > 0