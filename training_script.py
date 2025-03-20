from data.data_ingestion import DataIngestion
from data.data_processing import DataPreprocessor
from model.model_training import ModelTrainer
from imblearn.over_sampling import SMOTE
import logging
import argparse
import joblib
import os
import pandas as pd
import numpy as np

# Set up more verbose logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Also log to stdout for Kaggle notebook visibility
import sys
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

parser = argparse.ArgumentParser()
parser.add_argument("--sample_size", type=int, default=1000, help="Number of rows to sample from the dataset")
parser.add_argument("--use_dask", action="store_true", help="Whether to use Dask for data loading")
parser.add_argument("--no_dask", dest="use_dask", action="store_false", help="Disable Dask for data loading")
parser.add_argument("--chunk_size", type=int, default=50000, help="Size of chunks when reading large files")
parser.add_argument("--model_type", type=str, default="classifier", help="Type of model to train")
parser.add_argument("--use_ensemble", action="store_true", default=True, help="Whether to use ensemble models")
parser.add_argument("--no_ensemble", dest="use_ensemble", action="store_false", help="Disable ensemble models")
parser.add_argument("--ensemble_models", nargs="+", default=["xgboost", "lightgbm", "catboost", "rf"], 
                   help="List of ensemble models to use")
parser.add_argument("--tune_hyperparameters", action="store_true", default=True, help="Whether to tune hyperparameters")
parser.add_argument("--no_tuning", dest="tune_hyperparameters", action="store_false", help="Disable hyperparameter tuning")
parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
parser.add_argument("--experiment_name", type=str, default="debt_management", help="Name of the experiment")
parser.add_argument("--data_path", type=str, default="/kaggle/input/lending-club/accepted_2007_to_2018Q4.csv.gz", 
                   help="Path to the dataset")
parser.add_argument("--kaggle", action="store_true", help="Whether running on Kaggle")

args = parser.parse_args()

# Update cache_dir to use Kaggle working directory when on Kaggle
if args.kaggle:
    # Use Kaggle's working directory for output and cache
    output_dir = "/kaggle/working/models"
    cache_dir = "/kaggle/working/cache"
    logging.info("Running on Kaggle. Models will be saved to /kaggle/working/models")
    logging.info(f"Cache directory set to {cache_dir}")
else:
    # Use the default directories in the current folder
    output_dir = "models"
    cache_dir = "data/cache"
    logging.info(f"Models will be saved to {output_dir}")

# Configuration
DATA_CONFIG = {
    "cache_dir": cache_dir,
    "sample_size": args.sample_size,  
    "use_dask": args.use_dask if args.sample_size > 10000 else False,
    "chunk_size": args.chunk_size      
}

MODEL_CONFIG = {
    'models_dir': output_dir,
    'model_type': args.model_type,
    'use_ensemble': args.use_ensemble,
    'ensemble_models': args.ensemble_models,
    'tune_hyperparameters': args.tune_hyperparameters,
    'random_state': args.random_state,
    'experiment_name': args.experiment_name
}

logging.info(f"Running with configuration: sample_size={args.sample_size}, use_dask={DATA_CONFIG['use_dask']}, "
      f"use_ensemble={args.use_ensemble}, tune_hyperparameters={args.tune_hyperparameters}")

# Create data directories if they don't exist
os.makedirs(DATA_CONFIG["cache_dir"], exist_ok=True)
os.makedirs(MODEL_CONFIG["models_dir"], exist_ok=True)

logging.info("Starting data ingestion...")
data_ingestion = DataIngestion(config=DATA_CONFIG)

try:
    logging.info(f"Loading data from {args.data_path}...")
    data = data_ingestion.load_lending_club_data(
        file_path=args.data_path,  
        sample_size=DATA_CONFIG["sample_size"],
        use_cache=True
    )
    logging.info(f"Successfully loaded data with shape: {data.shape}")
except FileNotFoundError:
    logging.error(f"Error: {args.data_path} not found. Please check the file path.")
    exit(1)
except Exception as e:
    logging.error(f"Error loading data: {e}", exc_info=True)
    exit(1)

preprocessor = DataPreprocessor()

logging.info("Further cleaning and preprocessing data...")
preprocessed_data = preprocessor.clean_data(data)

# Manual handling of problematic columns
logging.info("Handling categorical variables and feature selection...")
# Convert term to numeric if it exists and is still a string
if 'term' in preprocessed_data.columns and preprocessed_data['term'].dtype == 'object':
    logging.info("Converting term column to numeric...")
    preprocessed_data['term'] = preprocessed_data['term'].str.extract('(\d+)').astype(float)

# Drop or convert other problematic string columns that can't be used by the model
object_columns = preprocessed_data.select_dtypes(include=['object']).columns
if len(object_columns) > 0:
    logging.info(f"Found {len(object_columns)} object columns: {list(object_columns)}")
    
    # For each categorical column, either encode it or drop it
    for col in object_columns:
        if col == 'target_column':
            continue  # Skip target column
            
        # If column has few unique values, encode it
        if preprocessed_data[col].nunique() < 20:
            logging.info(f"Encoding categorical column: {col} with {preprocessed_data[col].nunique()} values")
            # One-hot encode
            dummies = pd.get_dummies(preprocessed_data[col], prefix=col, drop_first=True)
            preprocessed_data = pd.concat([preprocessed_data, dummies], axis=1)
        else:
            logging.info(f"Dropping column with too many categories: {col}")
        
        # Remove original column after encoding/handling
        preprocessed_data = preprocessed_data.drop(columns=[col])

# Check for and handle any remaining non-numeric columns
non_numeric = preprocessed_data.select_dtypes(exclude=['number']).columns
if len(non_numeric) > 0 and 'target_column' not in non_numeric:
    logging.info(f"Dropping remaining non-numeric columns: {list(non_numeric)}")
    preprocessed_data = preprocessed_data.drop(columns=non_numeric)
elif 'target_column' in non_numeric:
    # Handle target column if needed
    logging.info("Converting target column to numeric")
    preprocessed_data['target_column'] = preprocessed_data['target_column'].astype(float)

# Convert all remaining columns to float for model compatibility
logging.info("Converting all columns to float...")
numeric_cols = preprocessed_data.columns.difference(['target_column'])
preprocessed_data[numeric_cols] = preprocessed_data[numeric_cols].astype(float)

# Check for NaN values and fill them
if preprocessed_data.isna().any().any():
    logging.info("Filling missing values...")
    preprocessed_data = preprocessed_data.fillna(preprocessed_data.mean())

# Check target variable distribution
target_counts = preprocessed_data['target_column'].value_counts()
logging.info(f"Target distribution: {target_counts}")

if len(target_counts) >= 2 and (target_counts.min() / target_counts.max() < 0.2):
    logging.info(f"Class imbalance detected. Original distribution {target_counts}")

    X = preprocessed_data.drop("target_column", axis = 1)
    y = preprocessed_data["target_column"]

    trainer = ModelTrainer(config = MODEL_CONFIG)
    X_train, y_train, X_test, y_test, X_val, y_val = trainer.split_data(preprocessed_data, "target_column", stratify=True)

    logging.info("Applying SMOTE to balance the training data...")
    smote = SMOTE(random_state = MODEL_CONFIG["random_state"])
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    logging.info(f"Training shape before SMOTE: {X_train.shape}")
    logging.info(f"Training shape after SMOTE: {X_train_resampled.shape}")
    logging.info(f"New class distribution in training: {pd.Series(y_train_resampled).value_counts()}")

    logging.info("Training model with balanced data")
    model = trainer.train_model(X_train_resampled, y_train_resampled, X_val, y_val)
else:
    logging.info("Class distribution is acceptable. Proceeding with regular training.")
    X, y = preprocessed_data.drop("target_column", axis = 1), preprocessed_data["target_column"]

    logging.info("Initializing model trainer...")
    trainer = ModelTrainer(config = MODEL_CONFIG)

    logging.info("Splitting data into train/validation/test sets...")
    X_train, y_train, X_test, y_test, X_val, y_val = trainer.split_data(preprocessed_data, "target_column", stratify=True)


logging.info("Training model...")
model = trainer.train_model(X_train, y_train, X_val, y_val)

logging.info("Evaluating model...")
metrics = trainer.evaluate_model(model, X_test, y_test)
logging.info(f"Model performance metrics: {metrics}")

# Save model and preprocessor
logging.info("Saving model and preprocessor...")
model_path = os.path.join(MODEL_CONFIG['models_dir'], 'best_model.pkl')
preprocessor_path = os.path.join(MODEL_CONFIG['models_dir'], 'preprocessor.pkl')

joblib.dump(model, model_path)
joblib.dump(preprocessor, preprocessor_path)

logging.info(f"Model saved to {model_path}")
logging.info(f"Preprocessor saved to {preprocessor_path}")

# If on Kaggle, print a message about downloading the model
if args.kaggle:
    logging.info("To download the model from Kaggle, look for the 'Output' tab and find best_model.pkl and preprocessor.pkl")
    # On Kaggle, you might also want to save to the root working directory for easier access
    root_model_path = "/kaggle/working/best_model.pkl"
    root_preprocessor_path = "/kaggle/working/preprocessor.pkl"
    joblib.dump(model, root_model_path)
    joblib.dump(preprocessor, root_preprocessor_path)
    logging.info(f"Also saved to {root_model_path} and {root_preprocessor_path} for easier access")

logging.info("Training complete!")