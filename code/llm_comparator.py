import requests
import json
import time
import os
import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LLMComparator:
    """Handles communication with multiple LLM APIs"""

    def __init__(self):
        self.api_config = {
            "openai": {
                "url": "https://api.openai.com/v1/chat/completions",
                "model": "gpt-4",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}",
                    "Content-Type": "application/json"
                }
            },
            "anthropic": {
                "url": "https://api.anthropic.com/v1/messages",
                "model": "claude-3-sonnet-20240229",
                "headers": {
                    "x-api-key": os.getenv('ANTHROPIC_API_KEY', ''),
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
            },
            "deepseek": {
                "url": "https://api.deepseek.com/v1/chat/completions",
                "model": "deepseek-chat",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY', '')}",
                    "Content-Type": "application/json"
                }
            }
        }

        self._validate_api_keys()

    def _validate_api_keys(self):
        """Validate if API keys are set"""
        for provider, config in self.api_config.items():
            if provider == "openai" and not os.getenv('OPENAI_API_KEY'):
                logger.warning(f"{provider.upper()} API key not set")
            elif provider == "anthropic" and not os.getenv('ANTHROPIC_API_KEY'):
                logger.warning(f"{provider.upper()} API key not set")
            elif provider == "deepseek" and not os.getenv('DEEPSEEK_API_KEY'):
                logger.warning(f"{provider.upper()} API key not set")

    def call_llm_api(self, prompt: str, provider: str = "openai", max_retries: int = 3) -> Optional[str]:
        """Call LLM API with retry mechanism"""
        if provider not in self.api_config:
            logger.error(f"Unsupported LLM provider: {provider}")
            return None

        if not self._check_api_availability(provider):
            logger.warning(f"{provider} API not available, using simulated response")
            return None

        for attempt in range(max_retries):
            try:
                if provider == "openai":
                    return self._call_openai(prompt)
                elif provider == "anthropic":
                    return self._call_anthropic(prompt)
                elif provider == "deepseek":
                    return self._call_deepseek(prompt)

            except Exception as e:
                logger.warning(f"{provider} API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        return None

    def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI API"""
        payload = {
            "model": self.api_config["openai"]["model"],
            "messages": [
                {
                    "role": "system",
                    "content": "You are an economic risk assessment expert. Based on Saaty's 1-9 scale, return only an integer between 1 and 9. No explanations, no additional text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 10
        }

        response = requests.post(
            self.api_config["openai"]["url"],
            headers=self.api_config["openai"]["headers"],
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"API returned error: {response.status_code} - {response.text}")

    def _call_anthropic(self, prompt: str) -> Optional[str]:
        """Call Anthropic API"""
        payload = {
            "model": self.api_config["anthropic"]["model"],
            "max_tokens": 10,
            "temperature": 0.1,
            "system": "You are an economic risk assessment expert. Based on Saaty's 1-9 scale, return only an integer between 1 and 9. No explanations, no additional text.",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        response = requests.post(
            self.api_config["anthropic"]["url"],
            headers=self.api_config["anthropic"]["headers"],
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result["content"][0]["text"].strip()
        else:
            raise Exception(f"API returned error: {response.status_code} - {response.text}")

    def _call_deepseek(self, prompt: str) -> Optional[str]:
        """Call DeepSeek API"""
        payload = {
            "model": self.api_config["deepseek"]["model"],
            "messages": [
                {
                    "role": "system",
                    "content": "You are an economic risk assessment expert. Based on Saaty's 1-9 scale, return only an integer between 1 and 9. No explanations, no additional text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 10
        }

        response = requests.post(
            self.api_config["deepseek"]["url"],
            headers=self.api_config["deepseek"]["headers"],
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(f"API returned error: {response.status_code} - {response.text}")

    def _check_api_availability(self, provider: str) -> bool:
        """Check if API is available"""
        if provider == "openai" and not os.getenv('OPENAI_API_KEY'):
            return False
        elif provider == "anthropic" and not os.getenv('ANTHROPIC_API_KEY'):
            return False
        elif provider == "deepseek" and not os.getenv('DEEPSEEK_API_KEY'):
            return False
        return True

    def parse_llm_response(self, response: str) -> int:
        """Parse LLM response to extract integer score"""
        if not response:
            return 5  # Default value

        try:
            # Extract numbers using regex
            numbers = re.findall(r'\b[1-9]\b', response)
            if numbers:
                return int(numbers[0])
            else:
                # Fallback logic based on keywords
                return self._fallback_score(response)
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return 5

    def _fallback_score(self, response: str) -> int:
        """Fallback scoring logic"""
        response_lower = response.lower()

        if any(word in response_lower for word in ['extremely', 'highly', 'much more', '9', 'nine']):
            return 9
        elif any(word in response_lower for word in ['very', 'strongly', '7', 'seven']):
            return 7
        elif any(word in response_lower for word in ['moderately', '5', 'five']):
            return 5
        elif any(word in response_lower for word in ['slightly', '3', 'three']):
            return 3
        elif any(word in response_lower for word in ['equal', '1', 'one']):
            return 1
        else:
            return 5