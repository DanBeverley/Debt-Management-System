from data.data_ingestion import DataIngestion
from data.data_processing import DataPreprocessor
from model.model_training import ModelTrainer
import argparse
import joblib
import os

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

args = parser.parse_args()

# Configuration
DATA_CONFIG = {
    "cache_dir": "data/cache",
    "sample_size": args.sample_size,  
    "use_dask": args.use_dask if args.sample_size > 10000 else False,
    "chunk_size": args.chunk_size      
}

MODEL_CONFIG = {
    'models_dir': 'models',
    'model_type': args.model_type,
    'use_ensemble': args.use_ensemble,
    'ensemble_models': args.ensemble_models,
    'tune_hyperparameters': args.tune_hyperparameters,
    'random_state': args.random_state,
    'experiment_name': args.experiment_name
}

print(f"Running with configuration: sample_size={args.sample_size}, use_dask={DATA_CONFIG['use_dask']}, "
      f"use_ensemble={args.use_ensemble}, tune_hyperparameters={args.tune_hyperparameters}")

# Create data directories if they don't exist
os.makedirs(DATA_CONFIG["cache_dir"], exist_ok=True)
os.makedirs(MODEL_CONFIG["models_dir"], exist_ok=True)

data_ingestion = DataIngestion(config=DATA_CONFIG)

print("Loading and preprocessing Lending Club data...")
try:
    data = data_ingestion.load_lending_club_data(
        file_path=args.data_path,  
        sample_size=DATA_CONFIG["sample_size"],
        use_cache=True
    )
    print(f"Successfully loaded data with shape: {data.shape}")
except FileNotFoundError:
    print(f"Error: {args.data_path} not found. Please check the file path.")
    exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

preprocessor = DataPreprocessor()

print("Further cleaning and preprocessing data...")
preprocessed_data = preprocessor.clean_data(data)

X, y = preprocessed_data.drop('target_column', axis=1), preprocessed_data['target_column']

print("Initializing model trainer...")
trainer = ModelTrainer(config=MODEL_CONFIG)

print("Splitting data into train/validation/test sets...")
X_train, y_train, X_test, y_test, X_val, y_val = trainer.split_data(
    preprocessed_data, 'target_column', stratify=True
)

print("Training model...")
model = trainer.train_model(X_train, y_train, X_val, y_val)

print("Evaluating model...")
metrics = trainer.evaluate_model(model, X_test, y_test)
print(f"Model performance metrics: {metrics}")

# Save model and preprocessor
print("Saving model and preprocessor...")
model_path = os.path.join(MODEL_CONFIG['models_dir'], 'best_model.pkl')
preprocessor_path = os.path.join(MODEL_CONFIG['models_dir'], 'preprocessor.pkl')

joblib.dump(model, model_path)
joblib.dump(preprocessor, preprocessor_path)

print(f"Model saved to {model_path}")
print(f"Preprocessor saved to {preprocessor_path}")
print("Training complete!")