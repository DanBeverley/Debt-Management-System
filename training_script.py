from data.data_ingestion import DataIngestion
from data.data_processing import DataPreprocessor
from model.model_training import ModelTrainer
import argparse
import joblib
import os
import pandas as pd
import numpy as np

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

# Manual handling of problematic columns
print("Handling categorical variables and feature selection...")
# Convert term to numeric if it exists and is still a string
if 'term' in preprocessed_data.columns and preprocessed_data['term'].dtype == 'object':
    print("Converting term column to numeric...")
    preprocessed_data['term'] = preprocessed_data['term'].str.extract('(\d+)').astype(float)

# Drop or convert other problematic string columns that can't be used by the model
object_columns = preprocessed_data.select_dtypes(include=['object']).columns
if len(object_columns) > 0:
    print(f"Found {len(object_columns)} object columns: {list(object_columns)}")
    
    # For each categorical column, either encode it or drop it
    for col in object_columns:
        if col == 'target_column':
            continue  # Skip target column
            
        # If column has few unique values, encode it
        if preprocessed_data[col].nunique() < 20:
            print(f"Encoding categorical column: {col} with {preprocessed_data[col].nunique()} values")
            # One-hot encode
            dummies = pd.get_dummies(preprocessed_data[col], prefix=col, drop_first=True)
            preprocessed_data = pd.concat([preprocessed_data, dummies], axis=1)
        else:
            print(f"Dropping column with too many categories: {col}")
        
        # Remove original column after encoding/handling
        preprocessed_data = preprocessed_data.drop(columns=[col])

# Check for and handle any remaining non-numeric columns
non_numeric = preprocessed_data.select_dtypes(exclude=['number']).columns
if len(non_numeric) > 0 and 'target_column' not in non_numeric:
    print(f"Dropping remaining non-numeric columns: {list(non_numeric)}")
    preprocessed_data = preprocessed_data.drop(columns=non_numeric)
elif 'target_column' in non_numeric:
    # Handle target column if needed
    print("Converting target column to numeric")
    preprocessed_data['target_column'] = preprocessed_data['target_column'].astype(float)

# Convert all remaining columns to float for model compatibility
print("Converting all columns to float...")
numeric_cols = preprocessed_data.columns.difference(['target_column'])
preprocessed_data[numeric_cols] = preprocessed_data[numeric_cols].astype(float)

# Check for NaN values and fill them
if preprocessed_data.isna().any().any():
    print("Filling missing values...")
    preprocessed_data = preprocessed_data.fillna(preprocessed_data.mean())

# Check target variable distribution
target_counts = preprocessed_data['target_column'].value_counts()
print(f"Target distribution: {target_counts}")

# If we have a severe class imbalance or only one class, create synthetic data
unique_classes = preprocessed_data['target_column'].nunique()
if unique_classes == 1:
    print("WARNING: Only one class found in target variable. Creating synthetic data for second class...")
    # Get the existing class
    existing_class = preprocessed_data['target_column'].iloc[0]
    # Create the opposite class (0 if existing is 1, 1 if existing is 0)
    opposite_class = 1.0 if existing_class == 0.0 else 0.0
    
    # Create a few synthetic rows with the opposite class
    # We'll use some of the existing rows but change the target value
    synthetic_rows = preprocessed_data.iloc[:10].copy()
    synthetic_rows['target_column'] = opposite_class
    
    # Add some random noise to the synthetic data to make it different
    for col in synthetic_rows.columns:
        if col != 'target_column' and synthetic_rows[col].dtype in [np.float64, np.float32, float]:
            # Add random noise to numeric columns
            synthetic_rows[col] = synthetic_rows[col] * (1 + np.random.uniform(-0.1, 0.1, size=len(synthetic_rows)))
    
    # Append the synthetic rows to the dataset
    preprocessed_data = pd.concat([preprocessed_data, synthetic_rows], ignore_index=True)
    print(f"Added {len(synthetic_rows)} synthetic rows with class {opposite_class}")
    print(f"New target distribution: {preprocessed_data['target_column'].value_counts()}")
elif target_counts.min() < 10:  # If the minority class has fewer than 10 samples
    print(f"WARNING: Severe class imbalance detected: {target_counts}")
    # We'll oversample the minority class a bit to help the model
    minority_class = target_counts.idxmin()
    minority_rows = preprocessed_data[preprocessed_data['target_column'] == minority_class]
    
    # Duplicate some minority class rows
    duplicates_needed = max(10, int(len(preprocessed_data) * 0.1)) - len(minority_rows)
    if duplicates_needed > 0:
        oversample_rows = minority_rows.sample(duplicates_needed, replace=True)
        # Add some noise to avoid exact duplicates
        for col in oversample_rows.columns:
            if col != 'target_column' and oversample_rows[col].dtype in [np.float64, np.float32, float]:
                oversample_rows[col] = oversample_rows[col] * (1 + np.random.uniform(-0.05, 0.05, size=len(oversample_rows)))
        
        preprocessed_data = pd.concat([preprocessed_data, oversample_rows], ignore_index=True)
        print(f"Added {len(oversample_rows)} rows to balance the minority class")
        print(f"New target distribution: {preprocessed_data['target_column'].value_counts()}")

print(f"Final preprocessed data shape: {preprocessed_data.shape}")

print("Separating features and target...")
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