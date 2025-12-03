# config.py
import os


class Config:
    """Configuration Class"""

    # API Settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')

    # Application Settings
    DEFAULT_LLM_PROVIDER = "openai"
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    REQUEST_DELAY = 1.0

    # Normalization Parameters
    NORMALIZATION_PARAMS = {
        "Inflation Rate": {"min": -2, "max": 15},
        "Unemployment Rate": {"min": 3, "max": 15},
        "GDP Growth Rate": {"min": -10, "max": 10},
        "Market Volatility": {"min": 0, "max": 100}
    }