class ExperimentEvaluator:
    """Class for evaluating experiment results"""

    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error"""
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
    def evaluate_performance(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> Dict:
        """Comprehensive performance evaluation"""
        target_columns = ["Recession_Probability", "Election_Probability", "Policy_Change_Probability"]

        rmse_scores = []
        cosine_scores = []

        for idx in range(len(ground_truth)):
            y_true = ground_truth[target_columns].iloc[idx].values
            y_pred = predictions[target_columns].iloc[idx].values

            rmse = ExperimentEvaluator.calculate_rmse(y_true, y_pred)
            cosine_sim = ExperimentEvaluator.calculate_cosine_similarity(y_true, y_pred)

            rmse_scores.append(rmse)
            cosine_scores.append(cosine_sim)

        return {
            "mean_rmse": np.mean(rmse_scores),
            "std_rmse": np.std(rmse_scores),
            "mean_cosine_similarity": np.mean(cosine_scores),
            "std_cosine_similarity": np.std(cosine_scores)
        }
