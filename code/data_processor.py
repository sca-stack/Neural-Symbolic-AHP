import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class EconomicDataProcessor:
    """Handles economic data processing and normalization"""

    def __init__(self):
        self.normalization_params = {
            "Inflation Rate": {"min": -2, "max": 15},
            "Unemployment Rate": {"min": 3, "max": 15},
            "GDP Growth Rate": {"min": -10, "max": 10},
            "Market Volatility": {"min": 0, "max": 100}
        }

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize economic data according to paper specifications"""
        df_normalized = df.copy()

        # Inflation Rate normalization
        if "Inflation Rate" in df.columns:
            min_val = self.normalization_params["Inflation Rate"]["min"]
            max_val = self.normalization_params["Inflation Rate"]["max"]
            df_normalized["Inflation Rate_normalized"] = (df["Inflation Rate"] - min_val) / (max_val - min_val)

        # Unemployment Rate normalization
        if "Unemployment Rate" in df.columns:
            min_val = self.normalization_params["Unemployment Rate"]["min"]
            max_val = self.normalization_params["Unemployment Rate"]["max"]
            df_normalized["Unemployment Rate_normalized"] = (df["Unemployment Rate"] - min_val) / (max_val - min_val)

        # GDP Growth Rate normalization
        if "GDP Growth Rate" in df.columns:
            min_val = self.normalization_params["GDP Growth Rate"]["min"]
            max_val = self.normalization_params["GDP Growth Rate"]["max"]
            df_normalized["GDP Growth Rate_normalized"] = (df["GDP Growth Rate"] - min_val) / (max_val - min_val)

        # Market Volatility normalization
        if "Market Volatility" in df.columns:
            max_val = self.normalization_params["Market Volatility"]["max"]
            df_normalized["Market Volatility_normalized"] = df["Market Volatility"] / max_val

        # Clip values to [0, 1] and handle missing values
        for col in df_normalized.columns:
            if 'normalized' in col:
                df_normalized[col] = df_normalized[col].fillna(0.5)  # Default to middle value
                df_normalized[col] = df_normalized[col].clip(0, 1)

        logger.info(f"Normalized data shape: {df_normalized.shape}")
        return df_normalized

    def generate_synthetic_data(self, n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
        """Generate synthetic economic data for testing"""
        np.random.seed(random_state)

        data = {
            "Inflation Rate": np.random.uniform(-2, 15, n_samples),
            "Unemployment Rate": np.random.uniform(3, 15, n_samples),
            "GDP Growth Rate": np.random.uniform(-10, 10, n_samples),
            "Market Volatility": np.random.uniform(0, 100, n_samples),
        }

        df = pd.DataFrame(data)
        df_normalized = self.normalize_data(df)

        logger.info(f"Generated {n_samples} synthetic economic data samples")
        return df_normalized

    def denormalize_data(self, df_normalized: pd.DataFrame) -> pd.DataFrame:
        """Convert normalized data back to original scale"""
        df_original = df_normalized.copy()

        # Inflation Rate denormalization
        if "Inflation Rate_normalized" in df_normalized.columns:
            min_val = self.normalization_params["Inflation Rate"]["min"]
            max_val = self.normalization_params["Inflation Rate"]["max"]
            df_original["Inflation Rate"] = df_normalized["Inflation Rate_normalized"] * (max_val - min_val) + min_val

        # Unemployment Rate denormalization
        if "Unemployment Rate_normalized" in df_normalized.columns:
            min_val = self.normalization_params["Unemployment Rate"]["min"]
            max_val = self.normalization_params["Unemployment Rate"]["max"]
            df_original["Unemployment Rate"] = df_normalized["Unemployment Rate_normalized"] * (
                        max_val - min_val) + min_val

        # GDP Growth Rate denormalization
        if "GDP Growth Rate_normalized" in df_normalized.columns:
            min_val = self.normalization_params["GDP Growth Rate"]["min"]
            max_val = self.normalization_params["GDP Growth Rate"]["max"]
            df_original["GDP Growth Rate"] = df_normalized["GDP Growth Rate_normalized"] * (max_val - min_val) + min_val

        # Market Volatility denormalization
        if "Market Volatility_normalized" in df_normalized.columns:
            max_val = self.normalization_params["Market Volatility"]["max"]
            df_original["Market Volatility"] = df_normalized["Market Volatility_normalized"] * max_val

        return df_original