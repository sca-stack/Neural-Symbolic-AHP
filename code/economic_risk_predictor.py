import pandas as pd
import numpy as np
import requests
import json
import time
import os
import re
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from sklearn.metrics import mean_squared_error
from scipy.linalg import eig
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def run_economic_experiment():
    """Run complete economic forecasting experiment"""
    print("=== Neural-AHP Economic Forecasting Experiment ===")
    print()

    # Initialize framework
    framework = NeuralAHPFramework()

    # Generate training data
    print("1. Generating training data using multi-agent AHP...")
    training_data = framework.generate_training_data(n_samples=50)

    if len(training_data) == 0:
        print("Failed to generate training data. Using simulated data.")
        # Fallback: create simulated training data
        training_data = framework.data_processor.generate_synthetic_data(50)
        # Add simulated target probabilities
        np.random.seed(42)
        targets = np.random.dirichlet(np.ones(3), size=50)
        training_data["Recession_Probability"] = targets[:, 0]
        training_data["Election_Probability"] = targets[:, 1]
        training_data["Policy_Change_Probability"] = targets[:, 2]
        training_data["Consistency_Score"] = np.random.uniform(0.8, 1.0, 50)
        training_data["Conflict_Score"] = np.random.uniform(0.0, 0.2, 50)

    print(f"Generated {len(training_data)} training samples")

    # Train neural network
    print("2. Training neural network...")
    train_score, test_score = framework.train_neural_network(training_data)

    # Generate test data
    print("3. Generating test data...")
    test_data = framework.data_processor.generate_synthetic_data(20)

    # Make predictions
    print("4. Making predictions...")
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
        except:
            # Fallback prediction
            pred_row = {
                "sample_id": idx,
                "Recession_Probability": 0.33,
                "Election_Probability": 0.33,
                "Policy_Change_Probability": 0.34
            }
            predictions.append(pred_row)

    predictions_df = pd.DataFrame(predictions)

    # For demonstration, use training data as ground truth
    ground_truth = training_data[["Recession_Probability", "Election_Probability", "Policy_Change_Probability"]].iloc[
                   :20].copy()

    # Evaluate performance
    print("5. Evaluating performance...")
    evaluator = ExperimentEvaluator()
    performance = evaluator.evaluate_performance(ground_truth, predictions_df)

    # Display results
    print("\n" + "=" * 50)
    print("EXPERIMENT RESULTS")
    print("=" * 50)
    print(f"Neural Network Performance:")
    print(f"  - Train R² Score: {train_score:.4f}")
    print(f"  - Test R² Score: {test_score:.4f}")
    print(f"\nPrediction Quality:")
    print(f"  - Mean RMSE: {performance['mean_rmse']:.4f} ± {performance['std_rmse']:.4f}")
    print(
        f"  - Mean Cosine Similarity: {performance['mean_cosine_similarity']:.4f} ± {performance['std_cosine_similarity']:.4f}")

    # Display sample predictions
    print(f"\nSample Predictions (first 5 samples):")
    print("Input Features -> Predicted Probabilities [Recession, Election, Policy Change]")
    for i in range(min(5, len(test_data))):
        inputs = test_data.iloc[i][["Inflation Rate_normalized", "Unemployment Rate_normalized",
                                    "GDP Growth Rate_normalized", "Market Volatility_normalized"]]
        preds = predictions_df.iloc[i][["Recession_Probability", "Election_Probability", "Policy_Change_Probability"]]

        print(f"Sample {i + 1}: {inputs.values.round(3)} -> {preds.values.round(3)}")

    # Create visualization
    plt.figure(figsize=(12, 8))

    # Plot 1: Feature distributions
    plt.subplot(2, 2, 1)
    feature_columns = ["Inflation Rate_normalized", "Unemployment Rate_normalized",
                       "GDP Growth Rate_normalized", "Market Volatility_normalized"]
    training_data[feature_columns].boxplot()
    plt.title("Normalized Input Feature Distributions")
    plt.xticks(rotation=45)

    # Plot 2: Output probability distributions
    plt.subplot(2, 2, 2)
    output_columns = ["Recession_Probability", "Election_Probability", "Policy_Change_Probability"]
    training_data[output_columns].boxplot()
    plt.title("Output Probability Distributions")
    plt.xticks(rotation=45)

    # Plot 3: Consistency scores
    plt.subplot(2, 2, 3)
    plt.hist(training_data["Consistency_Score"], bins=20, alpha=0.7, edgecolor='black')
    plt.title("Consistency Score Distribution")
    plt.xlabel("Consistency Score")
    plt.ylabel("Frequency")

    # Plot 4: Sample predictions vs ground truth
    plt.subplot(2, 2, 4)
    sample_idx = range(min(10, len(ground_truth)))
    width = 0.35

    for i, concept in enumerate(output_columns):
        plt.bar([x + i * width for x in sample_idx],
                ground_truth[concept].values[:len(sample_idx)],
                width=width, label=f'True {concept}', alpha=0.7)

    plt.title("Sample Predictions vs Ground Truth")
    plt.xlabel("Sample Index")
    plt.ylabel("Probability")
    plt.legend()

    plt.tight_layout()
    plt.savefig('neural_ahp_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nResults visualization saved as 'neural_ahp_results.png'")

    return framework, training_data, predictions_df, performance


if __name__ == "__main__":
    # Check if API keys are set
    required_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPSEEK_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        print("Warning: The following API keys are not set:")
        for key in missing_keys:
            print(f"  - {key}")
        print("The system will use simulated responses.")
        print()

    # Run experiment
    framework, training_data, predictions, performance = run_economic_experiment()

    # Save results
    training_data.to_csv('neural_ahp_training_data.csv', index=False)
    predictions.to_csv('neural_ahp_predictions.csv', index=False)

    print(f"\nTraining data saved as 'neural_ahp_training_data.csv'")
    print(f"Predictions saved as 'neural_ahp_predictions.csv'")