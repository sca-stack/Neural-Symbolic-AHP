import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import openai
import os
from tenacity import retry, stop_after_attempt, wait_exponential
from scipy.linalg import eig

logger = logging.getLogger(__name__)


class AHPAnalyzer:
    """Handles AHP matrix operations and priority calculations"""

    def __init__(self):
        # Random Index values for consistency calculation
        self.random_index = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
                             6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

    def create_reciprocal_matrix(self, comparisons: Dict[Tuple[str, str], float], concepts: List[str]) -> np.ndarray:
        """Create reciprocal matrix from pairwise comparisons"""
        n = len(concepts)
        matrix = np.ones((n, n))

        for (i, j), value in comparisons.items():
            idx_i = concepts.index(i)
            idx_j = concepts.index(j)
            matrix[idx_i, idx_j] = value
            matrix[idx_j, idx_i] = 1.0 / value

        logger.debug(f"Created reciprocal matrix:\n{matrix}")
        return matrix

    def calculate_priority_weights(self, matrix: np.ndarray) -> np.ndarray:
        """Calculate priority weights using eigenvalue decomposition"""
        try:
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = eig(matrix)

            # Find index of maximum real eigenvalue
            max_idx = np.argmax(np.real(eigenvalues))
            max_eigenvector = np.real(eigenvectors[:, max_idx])

            # Normalize eigenvector to get priority weights
            priority_weights = max_eigenvector / np.sum(max_eigenvector)

            # Ensure positive weights and normalize again
            priority_weights = np.abs(priority_weights)
            priority_weights = priority_weights / np.sum(priority_weights)

            logger.debug(f"Calculated priority weights: {priority_weights}")
            return priority_weights

        except Exception as e:
            logger.error(f"Eigenvalue decomposition failed: {e}")
            # Return uniform distribution as fallback
            n = matrix.shape[0]
            return np.ones(n) / n

    def calculate_consistency_ratio(self, matrix: np.ndarray) -> float:
        """Calculate consistency ratio for AHP matrix"""
        n = matrix.shape[0]

        if n <= 2:
            return 0.0  # 2x2 matrices are always consistent

        try:
            # Calculate consistency index
            eigenvalues, _ = eig(matrix)
            lambda_max = np.max(np.real(eigenvalues))
            ci = (lambda_max - n) / (n - 1)

            # Get random index
            ri = self.random_index.get(n, 1.49)

            # Calculate consistency ratio
            cr = ci / ri if ri > 0 else float('inf')

            logger.debug(f"Consistency Ratio: {cr:.4f} (CI: {ci:.4f}, RI: {ri:.4f})")
            return cr

        except Exception as e:
            logger.error(f"Consistency calculation failed: {e}")
            return float('inf')

    def validate_score(self, raw_score: float) -> float:
        """Validate and adjust LLM score using sigmoid function"""

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        validated_score = 1 + 8 * sigmoid(raw_score - 5)
        validated_score = max(1, min(9, validated_score))

        logger.debug(f"Validated score: {raw_score} -> {validated_score}")
        return validated_score

    def is_matrix_consistent(self, matrix: np.ndarray, threshold: float = 0.1) -> bool:
        """Check if AHP matrix is consistent based on threshold"""
        cr = self.calculate_consistency_ratio(matrix)
        return cr <= threshold


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
        return self.normalize_data(df)


class LLMComparator:
    """Handles communication with multiple LLM APIs"""

    def __init__(self):
        self.api_config = {
            "openai": {
                "model": "gpt-4",
            },
            "anthropic": {
                "model": "claude-3-sonnet-20240229",
            },
            "deepseek": {
                "model": "deepseek-chat",
            }
        }
        self._validate_api_keys()

    def _validate_api_keys(self):
        """Validate if API keys are set"""
        for provider in self.api_config.keys():
            if provider == "openai" and not os.getenv('OPENAI_API_KEY'):
                logger.warning(f"{provider.upper()} API key not set")
            elif provider == "anthropic" and not os.getenv('ANTHROPIC_API_KEY'):
                logger.warning(f"{provider.upper()} API key not set")
            elif provider == "deepseek" and not os.getenv('DEEPSEEK_API_KEY'):
                logger.warning(f"{provider.upper()} API key not set")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def call_llm_api(self, prompt: str, provider: str = "openai") -> Optional[str]:
        """Call LLM API with retry mechanism"""
        if provider not in self.api_config:
            logger.error(f"Unsupported LLM provider: {provider}")
            return None

        if not self._check_api_availability(provider):
            logger.warning(f"{provider} API not available, using simulated response")
            return self._generate_simulated_response(prompt)

        try:
            if provider == "openai":
                return self._call_openai(prompt)
            elif provider == "anthropic":
                return self._call_anthropic(prompt)
            elif provider == "deepseek":
                return self._call_deepseek(prompt)
        except Exception as e:
            logger.warning(f"{provider} API call failed: {e}")
            return self._generate_simulated_response(prompt)

    def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI API"""
        openai.api_key = os.getenv('OPENAI_API_KEY')

        response = openai.ChatCompletion.create(
            model=self.api_config["openai"]["model"],
            messages=[
                {
                    "role": "system",
                    "content": "You are an economic risk assessment expert. Based on Saaty's 1-9 scale, return only an integer between 1 and 9. No explanations, no additional text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=10
        )

        return response.choices[0].message.content.strip()

    def _call_anthropic(self, prompt: str) -> Optional[str]:
        """Call Anthropic API - placeholder implementation"""
        logger.warning("Anthropic API not implemented, using simulated response")
        return self._generate_simulated_response(prompt)

    def _call_deepseek(self, prompt: str) -> Optional[str]:
        """Call DeepSeek API - placeholder implementation"""
        logger.warning("DeepSeek API not implemented, using simulated response")
        return self._generate_simulated_response(prompt)

    def _check_api_availability(self, provider: str) -> bool:
        """Check if API is available"""
        if provider == "openai" and not os.getenv('OPENAI_API_KEY'):
            return False
        elif provider == "anthropic" and not os.getenv('ANTHROPIC_API_KEY'):
            return False
        elif provider == "deepseek" and not os.getenv('DEEPSEEK_API_KEY'):
            return False
        return True

    def _generate_simulated_response(self, prompt: str) -> str:
        """Generate simulated LLM response for testing"""
        # Extract concepts from prompt
        if "Recession" in prompt and "Election" in prompt:
            return "7"  # Recession is moderately more likely than Election
        elif "Recession" in prompt and "Policy Change" in prompt:
            return "5"  # Recession is equally to moderately more likely
        elif "Election" in prompt and "Policy Change" in prompt:
            return "3"  # Election is slightly more likely
        else:
            return "5"  # Default equal likelihood

    def parse_llm_response(self, response: str) -> int:
        """Parse LLM response to extract integer score"""
        if not response:
            return 5  # Default value

        try:
            # Extract numbers using regex
            import re
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


class NeuralAHPFramework:
    """Main Neural-AHP Framework class"""

    def __init__(self):
        self.llm_comparator = LLMComparator()
        self.ahp_analyzer = AHPAnalyzer()
        self.data_processor = EconomicDataProcessor()

        # Economic concepts
        self.input_concepts = ["Inflation Rate", "Unemployment Rate", "GDP Growth Rate", "Market Volatility"]
        self.output_concepts = ["Recession", "Election", "Policy Change"]

        # Cache for LLM responses
        self.comparison_cache = {}

        # Neural network components
        self.neural_network = None
        self.scaler = None
        self.is_trained = False

        logger.info("Neural-AHP Framework initialized successfully")

    def build_prompt(self, concept1: str, concept2: str, state_description: str) -> str:
        """Build standardized prompt for pairwise comparison"""
        return (
            f"Given current economic state: {state_description}, "
            f"compare the likelihood of occurrence of '{concept1}' over '{concept2}' "
            f"using Saaty's 1-9 scale (1=equal likelihood, 9=extremely more likely). "
            f"Return ONLY an integer between 1 and 9."
        )

    def get_state_description(self, normalized_values: List[float]) -> str:
        """Generate state description from normalized values"""
        inflation, unemployment, gdp_growth, volatility = normalized_values

        # Convert normalized values back to approximate real values for description
        inflation_real = inflation * (15 - (-2)) + (-2)
        unemployment_real = unemployment * (15 - 3) + 3
        gdp_growth_real = gdp_growth * (10 - (-10)) + (-10)
        volatility_real = volatility * 100

        inflation_desc = f"High Inflation={inflation_real:.1f}%"
        unemployment_desc = f"Unemployment Rate={unemployment_real:.1f}%"
        growth_desc = f"GDP Growth={gdp_growth_real:.1f}%"
        volatility_desc = f"Market Volatility={volatility_real:.1f}%"

        return f"[{inflation_desc}, {unemployment_desc}, {growth_desc}, {volatility_desc}]"

    def single_agent_ahp(self, normalized_inputs: List[float], llm_provider: str = "openai") -> Dict:
        """Single-agent AHP processing"""
        state_description = self.get_state_description(normalized_inputs)
        comparisons = {}

        # Get pairwise comparisons for all concept pairs
        for i in range(len(self.output_concepts)):
            for j in range(i + 1, len(self.output_concepts)):
                concept1 = self.output_concepts[i]
                concept2 = self.output_concepts[j]

                # Check cache first
                cache_key = f"{concept1}_{concept2}_{state_description}_{llm_provider}"
                if cache_key in self.comparison_cache:
                    score = self.comparison_cache[cache_key]
                    logger.debug(f"Using cached score for {cache_key}: {score}")
                else:
                    prompt = self.build_prompt(concept1, concept2, state_description)
                    response = self.llm_comparator.call_llm_api(prompt, llm_provider)
                    score = self.llm_comparator.parse_llm_response(response)

                    # Validate and store score
                    validated_score = self.ahp_analyzer.validate_score(score)
                    self.comparison_cache[cache_key] = validated_score
                    score = validated_score

                comparisons[(concept1, concept2)] = score

        # Create reciprocal matrix
        matrix = self.ahp_analyzer.create_reciprocal_matrix(comparisons, self.output_concepts)

        # Calculate priority weights
        weights = self.ahp_analyzer.calculate_priority_weights(matrix)

        # Calculate consistency ratio
        cr = self.ahp_analyzer.calculate_consistency_ratio(matrix)

        return {
            "priority_weights": weights,
            "consistency_ratio": cr,
            "comparison_matrix": matrix,
            "state_description": state_description
        }

    def multi_agent_ahp(self, normalized_inputs: List[float], agents: List[str] = None) -> Dict:
        """Multi-agent AHP with consensus mechanism"""
        if agents is None:
            agents = ["openai"]  # Use only OpenAI for simplicity in demo

        agent_results = {}
        consistency_scores = {}

        # Get results from each agent
        for agent in agents:
            try:
                result = self.single_agent_ahp(normalized_inputs, agent)
                agent_results[agent] = result["priority_weights"]
                consistency_scores[agent] = 1.0 / (result["consistency_ratio"] + 1e-6)
                logger.info(f"Agent {agent} completed: CR={result['consistency_ratio']:.4f}")
            except Exception as e:
                logger.warning(f"Agent {agent} failed: {e}")
                continue

        if not agent_results:
            raise Exception("All agents failed to produce results")

        # Calculate consistency-weighted consensus
        total_consistency = sum(consistency_scores.values())
        consensus_weights = np.zeros_like(list(agent_results.values())[0])

        for agent, weights in agent_results.items():
            weight_factor = consistency_scores[agent] / total_consistency
            consensus_weights += weight_factor * weights

        # Normalize consensus weights
        consensus_weights = consensus_weights / np.sum(consensus_weights)

        # Calculate conflict scores (standard deviation across agents)
        conflict_scores = {}
        for i, concept in enumerate(self.output_concepts):
            concept_weights = [weights[i] for weights in agent_results.values()]
            conflict_scores[concept] = np.std(concept_weights)

        return {
            "consensus_weights": consensus_weights,
            "conflict_scores": conflict_scores,
            "agent_results": agent_results,
            "consistency_scores": consistency_scores,
            "mean_consistency": np.mean(list(consistency_scores.values())),
            "mean_conflict": np.mean(list(conflict_scores.values()))
        }

    def generate_training_data(self, n_samples: int = 50, agents: List[str] = None) -> pd.DataFrame:
        """Generate training data using multi-agent AHP"""
        if agents is None:
            agents = ["openai"]  # Use only OpenAI for simplicity

        logger.info(f"Generating {n_samples} training samples using {len(agents)} agents")

        # Generate synthetic economic data
        synthetic_data = self.data_processor.generate_synthetic_data(n_samples)

        results = []
        successful_samples = 0

        for idx, row in synthetic_data.iterrows():
            try:
                normalized_inputs = [
                    row["Inflation Rate_normalized"],
                    row["Unemployment Rate_normalized"],
                    row["GDP Growth Rate_normalized"],
                    row["Market Volatility_normalized"]
                ]

                # Get multi-agent AHP results
                ahp_result = self.multi_agent_ahp(normalized_inputs, agents)

                result_row = {
                    "sample_id": idx,
                    "Inflation Rate": row["Inflation Rate_normalized"],
                    "Unemployment Rate": row["Unemployment Rate_normalized"],
                    "GDP Growth Rate": row["GDP Growth Rate_normalized"],
                    "Market Volatility": row["Market Volatility_normalized"],
                    "Recession_Probability": ahp_result["consensus_weights"][0],
                    "Election_Probability": ahp_result["consensus_weights"][1],
                    "Policy_Change_Probability": ahp_result["consensus_weights"][2],
                    "Consistency_Score": ahp_result["mean_consistency"],
                    "Conflict_Score": ahp_result["mean_conflict"]
                }

                results.append(result_row)
                successful_samples += 1

                # Progress logging
                if (successful_samples) % 10 == 0:
                    logger.info(f"Generated {successful_samples}/{n_samples} samples")

                # Add delay to avoid API rate limits
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue

        logger.info(f"Successfully generated {successful_samples} training samples")
        return pd.DataFrame(results)

    def train_neural_network(self, training_data: pd.DataFrame, test_size: float = 0.2):
        """Train neural network to learn AHP mappings"""
        logger.info("Training neural network...")

        # Prepare features and targets
        feature_columns = ["Inflation Rate", "Unemployment Rate", "GDP Growth Rate", "Market Volatility"]
        target_columns = ["Recession_Probability", "Election_Probability", "Policy_Change_Probability"]

        X = training_data[feature_columns].values
        y = training_data[target_columns].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create neural network based on paper specifications
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]
        hidden1 = max(8, 2 * input_dim)  # max(8, 2×input_dim)
        hidden2 = output_dim + input_dim // 2  # (output_dim + input_dim/2)

        self.neural_network = MLPRegressor(
            hidden_layer_sizes=(hidden1, hidden2),
            activation='tanh',
            random_state=42,
            max_iter=2000,
            learning_rate_init=0.001,
            early_stopping=True,
            n_iter_no_change=50
        )

        # Apply consistency-based weighting
        consistency_weights = training_data["Consistency_Score"].values
        train_indices = training_data.index[:len(X_train)]
        sample_weights = consistency_weights[train_indices]

        # Train model
        self.neural_network.fit(X_train_scaled, y_train)

        # Evaluate model
        train_score = self.neural_network.score(X_train_scaled, y_train)
        test_score = self.neural_network.score(X_test_scaled, y_test)

        # Calculate predictions for additional metrics
        y_pred = self.neural_network.predict(X_test_scaled)
        mse = np.mean((y_test - y_pred) ** 2)

        logger.info("Neural network training completed")
        logger.info(f"Network architecture: Input({input_dim}) -> Hidden({hidden1}, {hidden2}) -> Output({output_dim})")
        logger.info(f"Train R² score: {train_score:.4f}")
        logger.info(f"Test R² score: {test_score:.4f}")
        logger.info(f"Test MSE: {mse:.6f}")

        self.is_trained = True

        return {
            "train_score": train_score,
            "test_score": test_score,
            "test_mse": mse,
            "network_architecture": f"Input({input_dim})->Hidden({hidden1},{hidden2})->Output({output_dim})"
        }

    def predict(self, normalized_inputs: List[float]) -> np.ndarray:
        """Predict using trained neural network"""
        if not self.is_trained:
            raise Exception("Neural network not trained yet. Call train_neural_network() first.")

        X = np.array(normalized_inputs).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        predictions = self.neural_network.predict(X_scaled)[0]

        # Ensure probabilities sum to 1 and are non-negative
        predictions = np.maximum(predictions, 0)  # Ensure non-negative
        predictions = predictions / np.sum(predictions)  # Normalize to sum to 1

        return predictions

    def get_framework_info(self) -> Dict:
        """Get information about the framework state"""
        return {
            "is_trained": self.is_trained,
            "input_concepts": self.input_concepts,
            "output_concepts": self.output_concepts,
            "cache_size": len(self.comparison_cache),
            "neural_network": "Trained" if self.is_trained else "Not trained"
        }