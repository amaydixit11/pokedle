from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Tuple, Any, List

class BaseSolver(ABC):
    """Abstract base class for all solving algorithms"""
    
    def __init__(self, dataframe: pd.DataFrame, attributes: list):
        self.df = dataframe.copy()
        self.attributes = attributes
        self.feedback_history = []
    
    def add_feedback(self, guess_idx: int, feedback: Dict[str, str]):
        """
        Add feedback to history.
        
        Args:
            guess_idx: Index of the guessed Pokemon
            feedback: Dictionary of feedback for each attribute
        """
        self.feedback_history.append((guess_idx, feedback))
    
    @abstractmethod
    def next_guess(self) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Generate next guess.
        
        Returns:
            Tuple of (pokemon_series, info_dict)
        """
        pass
    
    @abstractmethod
    def update_feedback(self, guess: pd.Series, feedback: Dict[str, str]):
        """Update solver state with feedback from guess"""
        pass
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Get current state information for debugging/display.
        Default implementation - subclasses can override.
        """
        return {
            "feedback_count": len(self.feedback_history)
        }