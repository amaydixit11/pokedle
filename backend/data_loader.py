import pandas as pd
from typing import Optional

class DataLoader:
    """Singleton data loader for Pokemon dataset"""
    
    _instance = None
    _df = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataLoader, cls).__new__(cls)
        return cls._instance
    
    def load_data(self, filepath: str = "03_cleaned_with_images_and_evolutionary_stages.csv"):
        """Load Pokemon dataset"""
        if self._df is None:
            self._df = pd.read_csv(filepath)
            self._preprocess()
        return self._df
    
    def _preprocess(self):
        """Preprocess data"""
        # Convert numeric columns
        numeric_cols = ['Height', 'Weight', 'Generation']
        for col in numeric_cols:
            if col in self._df.columns:
                self._df[col] = pd.to_numeric(self._df[col], errors='coerce')
        
        # Handle missing values for Type2
        if 'Type2' in self._df.columns:
            # Keep NaN for Type2 as it's meaningful (single-type Pokemon)
            pass
        
        # Ensure image_url column exists
        if 'image_url' not in self._df.columns:
            self._df['image_url'] = ''
    
    def get_pokemon_by_name(self, name: str) -> Optional[pd.Series]:
        """Get Pokemon by name"""
        if self._df is None:
            return None
        
        matches = self._df[self._df['Original_Name'] == name]
        return matches.iloc[0] if not matches.empty else None
    
    def get_random_pokemon(self) -> pd.Series:
        """Get random Pokemon"""
        if self._df is None:
            raise ValueError("Dataset not loaded")
        return self._df.sample(1).iloc[0]
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get full dataframe"""
        if self._df is None:
            raise ValueError("Dataset not loaded")
        return self._df.copy()
    
    @property
    def pokemon_count(self) -> int:
        """Get total number of Pokemon"""
        return len(self._df) if self._df is not None else 0
    
    def get_pokemon_list(self) -> list:
        """Get list of all Pokemon names"""
        if self._df is None:
            return []
        return self._df['Original_Name'].tolist()
    
    def get_attribute_values(self, attribute: str) -> list:
        """Get all unique values for an attribute"""
        if self._df is None or attribute not in self._df.columns:
            return []
        
        values = self._df[attribute].dropna().unique().tolist()
        return sorted(values) if values else []