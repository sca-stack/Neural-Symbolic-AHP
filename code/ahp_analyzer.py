import numpy as np
from scipy.linalg import eig
from typing import Dict, List, Tuple
import logging

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
            eigenvalues, _ = eig(matrix)
            lambda_max = np.max(np.real(eigenvalues))
            ci = (lambda_max - n) / (n - 1)

            ri = self.random_index.get(n, 1.49)

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