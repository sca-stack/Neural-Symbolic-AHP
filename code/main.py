import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime
from typing import Dict

from neural_ahp_framework import NeuralAHPFramework

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neural_ahp_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExperimentEvaluator:
    """Class for evaluating experiment results"""

    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error"""
        from sklearn.metrics import mean_squared_error
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def calculate_cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Cosine Similarity"""
        dot_product = np.dot(y_true, y_pred)
        norm_true = np.linalg.norm(y_true)
        norm_pred = np.linalg.norm(y_pred)

        if norm_true == 0 or norm_pred == 0:
            return 0.0

        return dot_product / (norm_true * norm_pred)

    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def evaluate_performance(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> Dict:
        """Comprehensive performance evaluation"""
        target_columns = ["Recession_Probability", "Election_Probability", "Policy_Change_Probability"]

        rmse_scores = []
        cosine_scores = []
        mae_scores = []

        for idx in range(len(ground_truth)):
            y_true = ground_truth[target_columns].iloc[idx].values
            y_pred = predictions[target_columns].iloc[idx].values

            rmse = ExperimentEvaluator.calculate_rmse(y_true, y_pred)
            cosine_sim = ExperimentEvaluator.calculate_cosine_similarity(y_true, y_pred)
            mae = ExperimentEvaluator.calculate_mae(y_true, y_pred)

            rmse_scores.append(rmse)
            cosine_scores.append(cosine_sim)
            mae_scores.append(mae)

        return {
            "mean_rmse": np.mean(rmse_scores),
            "std_rmse": np.std(rmse_scores),
            "mean_cosine_similarity": np.mean(cosine_scores),
            "std_cosine_similarity": np.std(cosine_scores),
            "mean_mae": np.mean(mae_scores),
            "std_mae": np.std(mae_scores),
            "n_samples": len(ground_truth)
        }

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
        report.append(
            f"  - Cosine Similarity: {metrics['mean_cosine_similarity']:.4f} ± {metrics['std_cosine_similarity']:.4f}")
        report.append(f"  - RMSE: {metrics['mean_rmse']:.4f} ± {metrics['std_rmse']:.4f}")
        report.append(f"  - MAE: {metrics['mean_mae']:.4f} ± {metrics['std_mae']:.4f}")
        report.append(f"  - Samples Evaluated: {metrics['n_samples']}")

        report.append("\nINTERPRETATION:")
        if metrics['mean_cosine_similarity'] >= 0.9:
            report.append("  ✓ Excellent alignment with ground truth")
        elif metrics['mean_cosine_similarity'] >= 0.7:
            report.append("  ✓ Good alignment with ground truth")
        elif metrics['mean_cosine_similarity'] >= 0.5:
            report.append("  ~ Moderate alignment with ground truth")
        else:
            report.append("  ✗ Needs improvement in prediction accuracy")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


def setup_environment():
    """Setup environment and check dependencies"""
    print("=== Neural-AHP Economic Forecasting Experiment ===")
    print()

    # Check if API keys are set
    required_keys = ['OPENAI_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        print("Warning: The following API keys are not set:")
        for key in missing_keys:
            print(f"  - {key}")
        print("The system will use simulated responses for demonstration.")
        print()

    # Check required packages
    try:
        import sklearn
        import scipy
        import matplotlib
        print("✓ All required packages are available")
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Please install required packages from requirements.txt")
        return False

    return True


def generate_visualizations(training_data: pd.DataFrame, predictions: pd.DataFrame, metrics: Dict):
    """Generate comprehensive visualizations of results"""
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Input feature distributions
    plt.subplot(2, 3, 1)
    feature_columns = ["Inflation Rate", "Unemployment Rate", "GDP Growth Rate", "Market Volatility"]
    training_data[feature_columns].boxplot()
    plt.title("Normalized Input Feature Distributions")
    plt.xticks(rotation=45)
    plt.ylabel("Normalized Value")

    # Plot 2: Output probability distributions
    plt.subplot(2, 3, 2)
    output_columns = ["Recession_Probability", "Election_Probability", "Policy_Change_Probability"]
    training_data[output_columns].boxplot()
    plt.title("Output Probability Distributions (Training)")
    plt.xticks(rotation=45)
    plt.ylabel("Probability")

    # Plot 3: Consistency and Conflict scores
    plt.subplot(2, 3, 3)
    x_pos = np.arange(2)
    consistency_mean = training_data["Consistency_Score"].mean()
    conflict_mean = training_data["Conflict_Score"].mean()

    plt.bar(x_pos, [consistency_mean, conflict_mean], color=['green', 'orange'], alpha=0.7)
    plt.xticks(x_pos, ['Consistency', 'Conflict'])
    plt.title("Average Consistency and Conflict Scores")
    plt.ylabel("Score")
    plt.ylim(0, 1)

    # Add value labels on bars
    for i, v in enumerate([consistency_mean, conflict_mean]):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    # Plot 4: Sample predictions comparison
    plt.subplot(2, 3, 4)
    n_samples = min(8, len(predictions))
    sample_indices = range(n_samples)
    width = 0.25

    # Use training data as ground truth for demonstration
    ground_truth_samples = training_data.head(n_samples)

    for i, concept in enumerate(output_columns):
        true_values = ground_truth_samples[concept].values
        pred_values = predictions[concept].values[:n_samples]

        plt.bar([x + i * width for x in sample_indices], true_values,
                width=width, label=f'True {concept}', alpha=0.7)
        plt.bar([x + i * width for x in sample_indices], pred_values,
                width=width, label=f'Pred {concept}', alpha=0.4, linestyle='--')

    plt.xlabel("Sample Index")
    plt.ylabel("Probability")
    plt.title("Sample Predictions vs Ground Truth")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 5: Performance metrics
    plt.subplot(2, 3, 5)
    metric_names = ['Cosine Similarity', 'RMSE', 'MAE']
    metric_values = [
        metrics['mean_cosine_similarity'],
        metrics['mean_rmse'],
        metrics['mean_mae']
    ]

    colors = ['green', 'red', 'orange']
    bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.7)
    plt.title("Performance Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.4f}', ha='center', va='bottom')

    # Plot 6: Correlation heatmap of input features
    plt.subplot(2, 3, 6)
    correlation_matrix = training_data[feature_columns + output_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, cbar_kws={"shrink": 0.8})
    plt.title("Feature-Target Correlation Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig('neural_ahp_results.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_complete_experiment():
    """Run complete Neural-AHP experiment"""
    experiment_start = datetime.now()
    logger.info(f"Starting Neural-AHP experiment at {experiment_start}")

    # Initialize framework
    framework = NeuralAHPFramework()

    # Phase 1: Generate training data
    print("\n1. GENERATING TRAINING DATA")
    print("-" * 40)

    training_data = framework.generate_training_data(n_samples=30)  # Reduced for demo

    if len(training_data) == 0:
        print("Failed to generate training data. Creating simulated dataset...")
        # Create fallback simulated data
        training_data = framework.data_processor.generate_synthetic_data(30)
        # Add simulated target probabilities using economic logic
        np.random.seed(42)
        for idx in range(len(training_data)):
            inflation = training_data.iloc[idx]["Inflation Rate_normalized"]
            unemployment = training_data.iloc[idx]["Unemployment Rate_normalized"]
            gdp_growth = training_data.iloc[idx]["GDP Growth Rate_normalized"]

            # Simple economic logic for probabilities
            recession_prob = 0.3 + 0.4 * inflation + 0.2 * unemployment - 0.3 * gdp_growth
            election_prob = 0.2 + 0.5 * inflation + 0.1 * unemployment
            policy_prob = 0.5 - 0.2 * inflation + 0.3 * unemployment

            probs = np.array([recession_prob, election_prob, policy_prob])
            probs = np.maximum(probs, 0)
            probs = probs / np.sum(probs)

            training_data.loc[training_data.index[idx], "Recession_Probability"] = probs[0]
            training_data.loc[training_data.index[idx], "Election_Probability"] = probs[1]
            training_data.loc[training_data.index[idx], "Policy_Change_Probability"] = probs[2]

        training_data["Consistency_Score"] = np.random.uniform(0.8, 1.0, len(training_data))
        training_data["Conflict_Score"] = np.random.uniform(0.0, 0.2, len(training_data))

    print(f"✓ Generated {len(training_data)} training samples")

    # Phase 2: Train neural network
    print("\n2. TRAINING NEURAL NETWORK")
    print("-" * 40)

    training_results = framework.train_neural_network(training_data)
    print(f"✓ Neural network training completed")
    print(f"  - Train R2: {training_results['train_score']:.4f}")
    print(f"  - Test R2: {training_results['test_score']:.4f}")
    print(f"  - Architecture: {training_results['network_architecture']}")

    # Phase 3: Generate test data and predictions
    print("\n3. MAKING PREDICTIONS")
    print("-" * 40)

    test_data = framework.data_processor.generate_synthetic_data(15, random_state=123)
    predictions = []

    for idx, row in test_data.iterrows():
        normalized_inputs = [
            row["Inflation Rate_normalized"],
            row["Unemployment Rate_normalized"],
            row["GDP Growth Rate_normalized"],
            row["Market Volatility_normalized"]
        ]

        try:
            pred = framework.predict(normalized_inputs)
            pred_row = {
                "sample_id": idx,
                "Recession_Probability": pred[0],
                "Election_Probability": pred[1],
                "Policy_Change_Probability": pred[2]
            }
            predictions.append(pred_row)
        except Exception as e:
            logger.error(f"Prediction failed for sample {idx}: {e}")
            # Fallback uniform prediction
            pred_row = {
                "sample_id": idx,
                "Recession_Probability": 0.33,
                "Election_Probability": 0.33,
                "Policy_Change_Probability": 0.34
            }
            predictions.append(pred_row)

    predictions_df = pd.DataFrame(predictions)
    print(f"✓ Generated predictions for {len(predictions_df)} test samples")

    # Phase 4: Performance evaluation
    print("\n4. PERFORMANCE EVALUATION")
    print("-" * 40)

    # Use first 15 training samples as ground truth for demonstration
    ground_truth = training_data.head(15).copy()

    evaluator = ExperimentEvaluator()
    performance_metrics = evaluator.evaluate_performance(ground_truth, predictions_df)

    # Generate performance report
    framework_info = framework.get_framework_info()
    performance_report = evaluator.generate_performance_report(performance_metrics, framework_info)
    print(performance_report)


    experiment_end = datetime.now()
    duration = experiment_end - experiment_start

    logger.info(f"Experiment completed at {experiment_end}")
    logger.info(f"Total duration: {duration}")

    return framework, training_data, predictions_df, performance_metrics


def demonstrate_framework_capabilities(framework: NeuralAHPFramework):

    # Test with specific economic scenarios
    test_scenarios = [
        [0.9, 0.3, 0.2, 0.8],  # High inflation, low growth, high volatility
        [0.2, 0.7, 0.8, 0.3],  # Low inflation, high unemployment, good growth
        [0.5, 0.5, 0.5, 0.5],  # Moderate conditions
    ]

    scenario_descriptions = [
        "Crisis scenario (high inflation, slow growth)",
        "Stagflation risk (high unemployment, good growth)",
        "Moderate economic conditions"
    ]

    print("Framework predictions for different economic scenarios:")
    print()

    for i, (scenario, description) in enumerate(zip(test_scenarios, scenario_descriptions)):
        try:
            predictions = framework.predict(scenario)
            print(f"Scenario {i + 1}: {description}")
            print(f"  Input: Inflation={scenario[0]:.1f}, Unemployment={scenario[1]:.1f}, "
                  f"GDP Growth={scenario[2]:.1f}, Volatility={scenario[3]:.1f}")
            print(f"  Predictions: Recession={predictions[0]:.3f}, "
                  f"Election={predictions[1]:.3f}, Policy Change={predictions[2]:.3f}")
            print()
        except Exception as e:
            print(f"  Prediction failed: {e}")
            print()


if __name__ == "__main__":
    try:
        if setup_environment():
            framework, training_data, predictions, metrics = run_complete_experiment()
            demonstrate_framework_capabilities(framework)
        else:
            print("Environment setup failed. Please check the requirements.")
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        print(f"Experiment failed: {e}")
        raise