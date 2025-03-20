"""
Model prediction Module for AI Debt Management

uses the trained model to generate personalized debt management recommendations. Includes functions for
preprocessing user input data and predicting optimal debt repayment strategies
"""
import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Union, Optional
from sklearn.base import BaseEstimator

# Fix the TensorFlow import to handle both older and newer versions
try:
    import tensorflow as tf
except ImportError:
    tf = None

# If keras is not directly in tensorflow, try importing it separately
if tf is not None and not hasattr(tf, 'keras'):
    try:
        import keras
        tf.keras = keras
    except ImportError:
        logging.warning("Could not import keras. Neural network models will not be available.")
        
from data.data_processing import DataPreprocessor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers = [logging.FileHandler('model_prediction.log'),
                                logging.StreamHandler()])
logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self, model_path:str, preprocessor_path:str, config:Optional[Dict] = None):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.config = config or {}
        # Load the trained model
        self.model = self._load_model(model_path)
        self.data_preprocessor = joblib.load(preprocessor_path)

    def _load_model(self, model_path:str) -> Union[BaseEstimator, tf.keras.Model]:
        """
        Load the trained model from disk. Supports joblib (sklearn) and h5 (keras) format
        :param model_path:     Path to the trained model file
        :return:               Loaded model object
        """
        logger.info(f"Loading trained model from {model_path}")
        try:
            if model_path.endswith(".h5"):
                model = tf.keras.models.load_model(model_path)
            else:
                model = joblib.load(model_path)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def preprocess_user_data(self, user_data:Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Apply the same preprocessing steps to user input data as used in the training

        :param user_data:   Raw user input data
        :return:            preprocessed DataFrame ready for prediction
        """
        logger.info("Preprocessing user input data")

        if isinstance(user_data, dict):
            # Flatten nested structures (e.g. list of debts)
            user_data = self._flatten_user_data(user_data)
            user_data = pd.DataFrame([user_data])
        cleaned_data = self.data_preprocessor.clean_data(user_data)
        encoded_data = self.data_preprocessor.encode_categorical(cleaned_data, training = False)
        normalized_data = self.data_preprocessor.normalize_data(encoded_data, training = False)
        return normalized_data

    def _flatten_user_data(self, user_data:Dict) -> Dict:
        """
        Flatten nested structures in user data (e.g. list of debts) into features

        :param user_data:   Raw user input data with potential nested structure
        :return:            Flattened dictionary with aggregate features
        """
        flattened_data = user_data.copy()
        if "debts" in flattened_data and isinstance(flattened_data["debts"], list):
            debts = flattened_data.pop("debts")
            total_debt = sum(debt['amount'] for debt in debts)
            avg_interest_rate = np.mean([debt["interest_rate"] for debt in debts])
            min_payment_sum = sum(debt['minimum_payment'] for debt in debts)

            flattened_data["total_debt"] = total_debt
            flattened_data["avg_interest_rate"] = avg_interest_rate
            flattened_data["min_payment_sum"] = min_payment_sum

        return flattened_data

    def predict_debt_strategy(self, preprocessed_data:pd.DataFrame) -> Dict:
        """
        Use the trained model to predict the optimal debt repayment strategy

        :param preprocessed_data: Preprocessed user data
        :return:                  Recommendations for debt repayment strategy
        """
        logger.info("Predicting debt repayment strategy")
        # Make predictions using trained model
        if isinstance(self.model, tf.keras.Model):
            predictions = self.model.predict(preprocessed_data).flatten()
        elif hasattr(self.model, "predict"):
            predictions = self.model.predict(preprocessed_data)
        else:
            raise AttributeError("Model does not have a predict method")

        recommendations = self._generate_recommendations(predictions)
        return recommendations

    def _generate_recommendations(self, predictions:np.ndarray) -> Dict:
        """
        Generate debt repayment recommendations based on model predictions

        :param predictions:   Model predictions
        :return:              Debt repayment recommendations
        """
        model_type = self.config.get("model_type", "classifier")
        if model_type == "classifier":
            strategy = "pay_high_interest_first" if predictions[0] == 1 else "pay_low_balance_first"
            details = ("Focus on paying off high-interest debts first to minimize interest payments" if predictions[0] == 1 else
                       "Focus on paying off low-balance debts first to reduce number of debts")
        else:
            predicted_time = predictions[0]
            strategy = "optimize_repayment_time"
            details = f"Predicted repayment time is {predicted_time:.2f} months. Increase payments to shorted this."
        recommendations = {"strategy":strategy,
                           "details":details}
        logger.info(f"Generated recommendations: {recommendations}")
        return recommendations

    def save_recommendations(self, recommendations:Dict, output_path:str) -> None:
        """
        Save the recommendations to a JSON file

        :param recommendations:    Debt payment recommendations
        :param output_path:        Path to the output JSON file
        """
        logger.info(f"Saving recommendations to {output_path}")

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok = True)
            with open(output_path, "w") as f:
                json.dump(recommendations, f, indent = 4)
            logger.info("Recommendations saved successfully")
        except Exception as e:
            logger.error(f"Failed to save recommendations: {e}")
            raise

# Testing purpose
if __name__ == "__main__":
    # Define path to the trained model
    model_path = os.path.join("models", "trained_model.pkl")

    user_data = {"income":5000,
                 "expenses":2000,
                 "debts":[{"amount":10000, "interest_rate":0.15, "minimum_payment":200},
                          {"amount":5000, "interest_rate":0.10, "minimum_payment":100}], "credit_score":650}
    config = {"model_type": "classifier"}
    predictor = ModelPredictor(model_path, config = config)

    preprocessed_data = predictor.preprocess_user_data(user_data)
    recommendations = predictor.predict_debt_strategy(preprocessed_data)

    output_path = os.path.join("output", "recommendations.json")
    predictor.save_recommendations(recommendations, output_path)

    print(json.dumps(recommendations, indent = 4))
