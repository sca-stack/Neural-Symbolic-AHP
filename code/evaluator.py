import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class ExperimentEvaluator:
    """Class for evaluating experiment results and performance metrics"""
    
    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def calculate_cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Cosine Similarity between two vectors"""
        dot_product = np.dot(y_true, y_pred)
        norm_true = np.linalg.norm(y_true)
        norm_pred = np.linalg.norm(y_pred)
        
        if norm_true == 0 or norm_pred == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm_true * norm_pred)
        return max(0.0, min(1.0, cosine_sim))  # Ensure in [0, 1] range
    
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def calculate_kldivergence(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
        """Calculate KL Divergence between two probability distributions"""
        # Add epsilon to avoid log(0)
        y_true_safe = np.clip(y_true, epsilon, 1)
        y_pred_safe = np.clip(y_pred, epsilon, 1)
        
        # Renormalize
        y_true_safe = y_true_safe / np.sum(y_true_safe)
        y_pred_safe = y_pred_safe / np.sum(y_pred_safe)
        
        return np.sum(y_true_safe * np.log(y_true_safe / y_pred_safe))
    
    @staticmethod
    def evaluate_performance(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> Dict:
        """Comprehensive performance evaluation"""
        target_columns = ["Recession_Probability", "Election_Probability", "Policy_Change_Probability"]
        
        # Ensure both dataframes have the same samples
        common_indices = ground_truth.index.intersection(predictions.index)
        if len(common_indices) == 0:
            raise ValueError("No common samples between ground truth and predictions")
        
        ground_truth = ground_truth.loc[common_indices]
        predictions = predictions.loc[common_indices]
        
        rmse_scores = []
        cosine_scores = []
        mae_scores = []
        kl_scores = []
        
        for idx in common_indices:
            y_true = ground_truth.loc[idx, target_columns].values
            y_pred = predictions.loc[idx, target_columns].values
            
            rmse = ExperimentEvaluator.calculate_rmse(y_true, y_pred)
            cosine_sim = ExperimentEvaluator.calculate_cosine_similarity(y_true, y_pred)
            mae = ExperimentEvaluator.calculate_mae(y_true, y_pred)
            kl_div = ExperimentEvaluator.calculate_kldivergence(y_true, y_pred)
            
            rmse_scores.append(rmse)
            cosine_scores.append(cosine_sim)
            mae_scores.append(mae)
            kl_scores.append(kl_div)
        
        # Calculate overall metrics
        metrics = {
            "mean_rmse": np.mean(rmse_scores),
            "std_rmse": np.std(rmse_scores),
            "mean_cosine_similarity": np.mean(cosine_scores),
            "std_cosine_similarity": np.std(cosine_scores),
            "mean_mae": np.mean(mae_scores),
            "std_mae": np.std(mae_scores),
            "mean_kl_divergence": np.mean(kl_scores),
            "std_kl_divergence": np.std(kl_scores),
            "n_samples": len(common_indices)
        }
        
        logger.info(f"Performance evaluation completed on {len(common_indices)} samples")
        logger.info(f"Mean Cosine Similarity: {metrics['mean_cosine_similarity']:.4f}")
        logger.info(f"Mean RMSE: {metrics['mean_rmse']:.4f}")
        
        return metrics
    
    @staticmethod
    def generate_performance_report(metrics: Dict, framework_info: Dict) -> str:
        """Generate a comprehensive performance report"""
        report = []
        report.append("=" * 60)
        report.append("NEURAL-AHP FRAMEWORK PERFORMANCE REPORT")
        report.append("=" * 60)
        
        report.append("\nFRAMEWORK INFORMATION:")
        report.append(f"  - Trained: {framework_info['is_trained']}")
        report.append(f"  - Input Concepts: {', '.join(framework_info['input_concepts'])}")
        report.append(f"  - Output Concepts: {', '.join(framework_info['output_concepts'])}")
        report.append(f"  - Cache Size: {framework_info['cache_size']}")
        
        report.append("\nPERFORMANCE METRICS:")
        report.append(f"  - Cosine Similarity: {metrics['mean_cosine_similarity']:.4f} ¡À {metrics['std_cosine_similarity']:.4f}")
        report.append(f"  - RMSE: {metrics['mean_rmse']:.4f} ¡À {metrics['std_rmse']:.4f}")
        report.append(f"  - MAE: {metrics['mean_mae']:.4f} ¡À {metrics['std_mae']:.4f}")
        report.append(f"  - KL Divergence: {metrics['mean_kl_divergence']:.6f} ¡À {metrics['std_kl_divergence']:.6f}")
        report.append(f"  - Samples Evaluated: {metrics['n_samples']}")
        
        report.append("\nINTERPRETATION:")
        if metrics['mean_cosine_similarity'] >= 0.9:
            report.append("  ? Excellent alignment with ground truth")
        elif metrics['mean_cosine_similarity'] >= 0.7:
            report.append("  ? Good alignment with ground truth")
        elif metrics['mean_cosine_similarity'] >= 0.5:
            report.append("  ~ Moderate alignment with ground truth")
        else:
            report.append("  ? Needs improvement in prediction accuracy")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    @staticmethod
    def compare_with_baselines(neural_ahp_metrics: Dict, baseline_metrics: Dict) -> Dict:
        """Compare Neural-AHP performance with baseline methods"""
        comparison = {}
        
        for metric in ['mean_cosine_similarity', 'mean_rmse', 'mean_mae']:
            if metric in neural_ahp_metrics and metric in baseline_metrics:
                neural_ahp_val = neural_ahp_metrics[metric]
                baseline_val = baseline_metrics[metric]
                
                if 'cosine' in metric:
                    improvement = (neural_ahp_val - baseline_val) / baseline_val * 100
                else:  # For error metrics (RMSE, MAE)
                    improvement = (baseline_val - neural_ahp_val) / baseline_val * 100
                
                comparison[metric] = {
                    'neural_ahp': neural_ahp_val,
                    'baseline': baseline_val,
                    'improvement_percent': improvement
                }
        
        return comparison