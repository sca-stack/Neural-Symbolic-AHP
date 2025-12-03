class EconomicDataProcessor:
    """Economic data processing class"""

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
        min_val = self.normalization_params["Inflation Rate"]["min"]
        max_val = self.normalization_params["Inflation Rate"]["max"]
        df_normalized["Inflation Rate_normalized"] = (df["Inflation Rate"] - min_val) / (max_val - min_val)

        # Unemployment Rate normalization
        min_val = self.normalization_params["Unemployment Rate"]["min"]
        max_val = self.normalization_params["Unemployment Rate"]["max"]
        df_normalized["Unemployment Rate_normalized"] = (df["Unemployment Rate"] - min_val) / (max_val - min_val)

        # GDP Growth Rate normalization
        min_val = self.normalization_params["GDP Growth Rate"]["min"]
        max_val = self.normalization_params["GDP Growth Rate"]["max"]
        df_normalized["GDP Growth Rate_normalized"] = (df["GDP Growth Rate"] - min_val) / (max_val - min_val)

        # Market Volatility normalization
        max_val = self.normalization_params["Market Volatility"]["max"]
        df_normalized["Market Volatility_normalized"] = df["Market Volatility"] / max_val

        # Clip values to [0, 1]
        for col in df_normalized.columns:
            if 'normalized' in col:
                df_normalized[col] = df_normalized[col].clip(0, 1)

        return df_normalized

    def generate_synthetic_data(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate synthetic economic data for testing"""
        np.random.seed(42)

        data = {
            "Inflation Rate": np.random.uniform(-2, 15, n_samples),
            "Unemployment Rate": np.random.uniform(3, 15, n_samples),
            "GDP Growth Rate": np.random.uniform(-10, 10, n_samples),
            "Market Volatility": np.random.uniform(0, 100, n_samples),
        }

        df = pd.DataFrame(data)
        return self.normalize_data(df)
