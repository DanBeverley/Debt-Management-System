"""
Model Training Module for AI Debt Management System

Handles the training, evaluation, and persistence of machine learning models for debt management prediction tasks.
"""
import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Union, Optional, Any, cast
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score, log_loss)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# For model interpretability
try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
# For model persistence
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.xgboost
import mlflow.lightgbm

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                   handlers = [logging.FileHandler('model_training.log'),
                                                       logging.StreamHandler()])
logger = logging.getLogger(__name__)

class ModelTrainer:
         def __init__(self, config : Optional[Dict] = None):
               self.config = config or {}
               expected_keys = {
                   'models_dir': str, 'random_state': int, 'n_jobs': int, 'cv_folds': int,
                   'test_size': float, 'val_size': float, 'mlflow_tracking_uri': str,
                   'model_type': str, 'model_algorithm': str, 'tune_hyperparameters': bool,
                   'use_mlflow': bool, 'use_ensemble': bool, 'ensemble_models': list,
                   'experiment_name': str
               }
               for key, value in self.config.items():
                     if key not in expected_keys:
                         logger.warning(f"Unexpected config key: {key}")
                     elif not isinstance(value, expected_keys.get(key, object)):
                         raise ValueError(f"Invalid type for {key}: expected {expected_keys[key]}, got {type(value)}")

               # Default config if not provided
               self.models_dir = self.config.get("models_dir", "models")
               self.random_state = self.config.get("random_state", 42)
               self.n_jobs = self.config.get("n_jobs", -1)
               self.cv_folds = self.config.get("cv_folds", 5)
               self.test_size  = self.config.get("test_size", 0.2)
               self.val_size = self.config.get("val_size", 0.15)
               self.tracking_uri = self.config.get("mlflow_tracking_uri", "mlruns")
               # Model configurations
               self.model_type = self.config.get('model_type', "classifier")
               self.model_algorithm = self.config.get('model_algorithm', 'ensemble')
               self.tune_hyperparameters = self.config.get('tune_hyperparameters', True)
               self.use_mlflow = self.config.get("use_mlflow", True)
               # Ensemble model configuration
               self.use_ensemble = self.config.get("use_ensemble", True)
               self.ensemble_models = self.config.get("ensemble_models", ['xgboost', 'lightgbm', 'catboost'])
               # Create model directory if it doesn't exist
               os.makedirs(self.models_dir, exist_ok = True)
               # Initialize variables
               self.model = None
               self.feature_importance = None
               self.trained = False
               self.X_train = None
               self.Y_train = None
               self.X_test = None
               self.Y_test = None
               self.X_val = None
               self.Y_val = None
               self.training_time = None
               self.metrics = {}

               if self.use_mlflow:
                   mlflow.set_tracking_uri(self.tracking_uri)
                   experiment_name = self.config.get("experiment_name", "debt_management")
                   try:
                       self.experiment_id = mlflow.create_experiment(experiment_name)
                   except:
                       self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

         def split_data(self, data:pd.DataFrame, target_col:str,
                                test_size:Optional[float] = None,
                                val_size:Optional[float] = None,
                                stratify:bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
                """
                Splits the dataset into training, validation, and testing sets

                :param data:                   Preprocessed dataset
                :param target_col:          Name of the target column
                :param test_size:             Proportion of data for testing (default from config)
                :param val_size:              Proportion of data for validation (default from config)
                :param stratify:               Whether to use stratified sampling for classification tasks
                :return:                            (X_train, y_train, X_test, y_test, X_val, y_val)
                """
                logger.info("Splitting data into train, validation, and test sets")
                if target_col not in data.columns:
                    raise ValueError(f"Target column '{target_col} not found in data'")

                test_size = test_size if test_size is not None else self.test_size
                val_size = val_size if val_size is not None else self.val_size
                # Separate features and target
                X = data.drop(columns = [target_col])
                y = data[target_col]
                # Determine if we should use stratification
                stratify_param = y if stratify and self.model_type == "classifier" else None
                if val_size > 0:
                    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = test_size, random_state = self.random_state,
                                                                                                               stratify = stratify_param)
                    stratify_param_val = y_train_val if stratify and self.model_type == "classifier" else None
                    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = val_size/(1 - test_size),
                                                                                                  random_state = self.random_state,
                                                                                                  stratify = stratify_param_val)
                    logger.info(f"Data split: Train={X_train.shape}, Validation={X_val.shape}, Test={X_test.shape}")
                    # Save the split for later use
                    self.X_train, self.y_train = X_train, y_train
                    self.X_test, self.y_test = X_test, y_test
                    self.X_val, self.y_val = X_val, y_val
                    return X_train, y_train, X_test, y_test, X_val, y_val
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = self.random_state,
                                                                                                  stratify = stratify_param)
                    logger.info(f"Data split (no validation): Train={X_train.shape}, Test={X_test.shape}")
                    # Save the splits for later use
                    self.X_train, self.y_train = X_train, y_train
                    self.X_test, self.y_test = X_test, y_test
                    self.X_val, self.y_val = None, None
                    return X_train, y_train, X_test, y_test, None, None

         def _create_stacked_ensemble(self, task_type:str) -> Union[StackingClassifier, Any]:
                  """
                  Create a stacked ensemble model combining multiple algorithms

                  :param task_type:  "classifier" or "regressor"
                  :return:                   A sklearn compatible stacked ensemble model
                  """
                  estimators = []
                  if task_type == "classifier":
                      if 'xgboost' in self.ensemble_models:
                          xgb_model = xgb.XGBClassifier(learning_rate=0.05, n_estimators=300, max_depth = 5,
                                                                                min_child_weight = 3, subsample = 0.8, colsample_bytree = 0.8,
                                                                                objective = 'binary:logistic', n_jobs = self.n_jobs, random_state = self.random_state)
                          estimators.append(('xgb', xgb_model))
                      if "lightgbm" in self.ensemble_models:
                          lgb_model = lgb.LGBMClassifier(learning_rate = 0.05, n_estimators = 300, max_depth = 5,
                                                                                 subsample = 0.8, colsample_bytree = 0.8, n_jobs = self.n_jobs,
                                                                                 random_state = self.random_state)
                          estimators.append(('lgb', lgb_model))
                      if 'catboost' in self.ensemble_models:
                          cb_model = cb.CatBoostClassifier(
                              learning_rate=0.05, iterations=300, depth=5, verbose=False,
                              subsample=0.8, colsample_bylevel=0.8, random_seed=self.random_state
                          )
                          estimators.append(('cb', cb_model))

                      if 'rf' in self.ensemble_models or len(estimators) == 0:
                          rf_model = RandomForestClassifier(
                              n_estimators=300, max_depth=10, random_state=self.random_state, n_jobs=self.n_jobs
                          )
                          estimators.append(('rf', rf_model))

                      if HAS_TENSORFLOW and 'nn' in self.ensemble_models:
                           def create_keras_model():
                                  model = Sequential([
                                      Dense(128, input_shape = (self.X_train.shape[1], ), activation = "relu"),
                                      BatchNormalization(),
                                      Dropout(0.3),
                                      Dense(64, activation = "relu"),
                                      BatchNormalization(),
                                      Dropout(0.3),
                                      Dense(32, activation="relu"),
                                      Dense(1, activation="sigmoid")
                                  ])
                                  model.compile(optimizer = Adam(learning_rate = 3e-4),
                                                            loss = "binary_crossentropy",
                                                            metrics = ["accuracy"])
                                  return model

                           nn_model = KerasClassifier(build_fn = create_keras_model,
                                                                          epochs = 50, batch_size = 64, verbose = 0)
                           estimators.append(("nn", nn_model))

                      if not estimators:
                          raise ValueError("No valid models specified in ensemble_models for classifier")

                      # Simple logistic regression as final estimator
                      final_estimator = LogisticRegression(max_iter = 1000, random_state = self.random_state)
                      return StackingClassifier(estimators = estimators, final_estimator = final_estimator,
                                                                 cv = 5, n_jobs= self.n_jobs, passthrough=True)
                  elif task_type == "regressor":
                      if 'xgboost' in self.ensemble_models:
                          estimators.append(('xgb', xgb.XGBRegressor(
                              learning_rate=0.05, n_estimators=300, max_depth=5,
                              min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                              n_jobs=self.n_jobs, random_state=self.random_state
                          )))
                      if 'lightgbm' in self.ensemble_models:
                          estimators.append(('lgb', lgb.LGBMRegressor(
                              learning_rate=0.05, n_estimators=300, max_depth=5,
                              subsample=0.8, colsample_bytree=0.8, n_jobs=self.n_jobs,
                              random_state=self.random_state
                          )))
                      if 'catboost' in self.ensemble_models:
                          estimators.append(('cb', cb.CatBoostRegressor(
                              learning_rate=0.05, iterations=300, depth=5, verbose=0,
                              subsample=0.8, colsample_bylevel=0.8, random_seed=self.random_state
                          )))
                      if 'rf' in self.ensemble_models:
                          estimators.append(('rf', RandomForestRegressor(
                              n_estimators=300, max_depth=10, random_state=self.random_state, n_jobs=self.n_jobs
                          )))
                      if HAS_TENSORFLOW and 'nn' in self.ensemble_models:
                          def create_keras_regressor():
                              model = Sequential([
                                  Dense(128, input_shape=(self.X_train.shape[1],), activation='relu'),
                                  BatchNormalization(),
                                  Dropout(0.3),
                                  Dense(64, activation='relu'),
                                  BatchNormalization(),
                                  Dropout(0.3),
                                  Dense(32, activation='relu'),
                                  Dense(1, activation='linear')
                              ])
                              model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
                              return model

                          estimators.append(
                              ('nn', KerasRegressor(build_fn=create_keras_regressor, epochs=50, batch_size=64, verbose=0)))

                      if not estimators:
                          raise ValueError("No valid models specified in ensemble_models for regressor")

                      final_estimator = LinearRegression()
                      return StackingRegressor(
                          estimators=estimators, final_estimator=final_estimator, cv=5, n_jobs=self.n_jobs, passthrough=True
                      )

                  else:
                      raise ValueError(f"Unsupported task_type: {task_type}. Use 'classifier' or 'regressor'.")

         def train_model(self, X_train:Optional[pd.DataFrame] = None, y_train:Optional[pd.Series] = None,
                                   X_val:Optional[pd.DataFrame] = None, y_val:Optional[pd.DataFrame] = None,
                                   model_params:Optional[Dict] = None) -> Any:
               """
               Train a machine learning model using preprocessed training data

               :param X_train:                Features for training
               :param y_train:                 Target for training
               :param X_val:                    Features for validation
               :param y_val:                    Target for validation
               :param model_params:    model hyperparameters
               :return:                               Trained model object
               """
               X_train = X_train if X_train is not None else self.X_train
               Y_train = y_train if y_train is not None else self.y_train
               X_val = X_val if X_val is not None else self.X_val
               y_val = y_val if y_val is not None else self.y_val

               if X_train is None and y_train is None:
                   raise ValueError("Training data not provided. Call split_data() first or provide X_train and y_train")
               logger.info(f"Starting model training with {self.model_algorithm} algorithm")
               start_time = time.time()

               if self.use_ensemble:
                   logger.info("Using stacked ensemble model for improved performance")
                   model = self._create_stacked_ensemble(self.model_type)
               else:
                   if self.model_algorithm == "xgboost":
                       if self.model_type == "classifier":
                           model = xgb.XGBClassifier(n_estimators = 300, max_depth = 5, learning_rate = 3e-4,
                                                                         subsample = 0.8, colsample_bytree = 0.8, random_state = self.random_state,
                                                                         n_jobs = self.n_jobs)
                       else:
                           model = xgb.XGBRegressor(
                               n_estimators=300, max_depth=5, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=self.random_state, n_jobs=self.n_jobs
                           )
                   elif self.model_algorithm == 'lightgbm':
                           if self.model_type == 'classifier':
                               model = lgb.LGBMClassifier(
                                   n_estimators=300, max_depth=5, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=self.random_state, n_jobs=self.n_jobs
                               )
                           else:
                               model = lgb.LGBMRegressor(
                                   n_estimators=300, max_depth=5, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8,
                                   random_state=self.random_state, n_jobs=self.n_jobs
                               )
                   else:
                           # Default to RandomForest
                           if self.model_type == 'classifier':
                               model = RandomForestClassifier(
                                   n_estimators=300, max_depth=10,
                                   random_state=self.random_state, n_jobs=self.n_jobs
                               )
                           else:
                               model = RandomForestRegressor(
                                   n_estimators=300, max_depth=10,
                                   random_state=self.random_state, n_jobs=self.n_jobs
                               )
               if model_params:
                  model.set_params(**model_params)
               if self.tune_hyperparameters and not isinstance(model, (StackingClassifier, StackingRegressor)):
                   logger.info("Tuning hyperparameters")
                   model = self._tune_hyperparameters(model, X_train, y_train, X_val, y_val)
               if self.use_mlflow:
                   with mlflow.start_run(experiment_id=self.experiment_id):
                       mlflow.log_params(model.get_params())
                       if HAS_TENSORFLOW and isinstance(model, (KerasClassifier, KerasRegressor)):
                           callbacks = [EarlyStopping(monitor='val_loss', patience = 10, restore_best_weights= True),
                                                ReduceLROnPlateau(monitor="val_loss", factor = 0.2, patience = 5, min_lr = 1e-6)]
                           keras_model = model.model_
                           history = keras_model.fit(
                               X_train, y_train,
                               validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                               validation_split=0.2 if (X_val is None or y_val is None) else 0.0,
                               callbacks=callbacks,
                               epochs = model.epochs,
                               batch_size = model.batch_size,
                               verbose = model.verbose
                           )
                           model.model_ = keras_model
                           # Log the final value of each metric in history
                           for metric_name, metric_values in history.history.items():
                               mlflow.log_metric(metric_name, metric_values[-1])
                       else:
                           model.fit(X_train, y_train)
                       mlflow.log_metric("training_time", time.time() - start_time)
                       mlflow.sklearn.log_model(model, "model")
               else:
                   if HAS_TENSORFLOW and isinstance(model, (KerasClassifier, KerasRegressor)):
                       callbacks = [
                           EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                           ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
                       ]
                       keras_model = model.model_
                       keras_model.fit(
                           X_train, y_train,
                           validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                           validation_split=0.2 if (X_val is None or y_val) is None else 0.0,
                           callbacks=callbacks
                       )
                       model.model_ = keras_model
                   else:
                       model.fit(X_train, y_train)
               self.training_time = time.time() - start_time
               logger.info(f"Model training completed in {self.training_time:.2f} seconds")

               self._extract_feature_importance(model, X_train)
               self.model = model
               self.trained = True
               return model

         def _tune_hyperparameters(self, model, X_train, y_train, X_val, y_val, n_trials:int = 50) -> Any:
                 """
                 Tunes hyperparameters using Optuna

                 :param model:    model for tuning
                 :param X_train:    training features
                 :param y_train:     training target
                 :param X_val:        validation features
                 :param y_val:         validation targets
                 :param n_trials:     number of tuning rials
                 :return:                   model with optimized parameters
                 """
                 if not hasattr(model, "set_params"):
                     logger.warning("Model does not support set_params, Skipping hyperparameters tuning")
                     return model
                 def objective(trial):
                     """Objective function for hyperparameter optimization"""
                     if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
                         params = {
                             'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                             'max_depth': trial.suggest_int('max_depth', 3, 10),
                             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                             'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                         }
                     elif isinstance(model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
                         params = {
                             'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                             'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                             'max_depth': trial.suggest_int('max_depth', 3, 10),
                             'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                             'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                         }
                     elif isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
                         params = {
                             'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                             'max_depth': trial.suggest_int('max_depth', 3, 15),
                             'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                             'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                         }
                     else:
                         # Generic model - skip tuning
                         return 0.0
                     # Create temporary model with new parameters
                     temp_model = cast(BaseEstimator, clone_model(model))
                     temp_model.set_params(**params)
                     # Cross validation for evaluation
                     if self.model_type == "classifier":
                         score = cross_val_score(temp_model, X_train, y_train, cv=3, scoring = "roc_auc", n_jobs = self.n_jobs)
                     else:
                         score = -1 * cross_val_score(temp_model, X_train, y_train, cv = 3, scoring = "neg_mean_squared_error",
                                                                        n_jobs = self.n_jobs).mean()
                     return score

                 from sklearn.base import clone as clone_model
                 # Set up Optuna
                 study_name = f"{model.__class__.__name__}_optimization"
                 sampler = TPESampler(seed = self.random_state)
                 pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
                 try:
                     study = optuna.create_study(study_name = study_name,
                                                                       direction="maximize" if self.model_type == "classifier" else "minimize",
                                                                       sampler = sampler, pruner = pruner)
                     study.optimize(objective, n_trials = n_trials)
                     best_params = study.best_params
                     logger.info(f"Best hyperparameters: {best_params}")

                     optimized_model = cast(BaseEstimator, clone_model(model))
                     optimized_model.set_params(**best_params)

                     return optimized_model
                 except Exception as e:
                     logger.warning(f"Hyperparameter tuning failed {e}. Using original model")
                     return model

         def _extract_feature_importance(self, model, X:pd.DataFrame) -> None:
               """
               Extract feature importance from the model if available

               :param model:    Trained model
               :param X:             Feature DataFrame to get column names
               """
               try:
                   if hasattr(model, "feature_importances_"):
                       # For tree-based model (RandomForest, XGBoost, etc)
                       importance = model.feature_importances_
                   elif hasattr(model, "coef_"):
                       # Linear models
                       importance = np.abs(model.coef_).flatten()
                   elif hasattr(model, "feature_importances_"):
                       # LightBGM models
                       importance = model.feature_importances_()
                   elif isinstance(model, StackingClassifier) or isinstance(model, StackingRegressor):
                       # For ensemble models, get importance from the final estimator
                       if hasattr(model.final_estimator_, "coef_"):
                           importance = np.abs(model.final_estimator_.coef_).flatten()
                           feature_names = [f"feature_{i}" for i in range(len(importance))]
                           self.feature_importance = dict(zip(feature_names, importance))
                           return
                       else:
                           # Can't extract from this ensemble
                           return
                   else:
                           # Model doesn't support built-in feature importance
                           return
                   # Map importance to feature names
                   feature_names = X.columns
                   self.feature_importance = dict(zip(feature_names, importance))
                   # Create log for most important features
                   top_features = sorted(self.feature_importance.items(), key = lambda x:x[1], reverse = True)[:10]
                   logger.info("Top 10 features by importance: ")
                   for feature, importance_val in top_features:
                       logger.info(f"   {feature}   :   {importance_val:.4f}")
               except Exception as e:
                   logger.warning(f"Could not extract feature importance: {e}")

         def evaluate_model(self, model: Optional[Any] = None,
                            X_test: Optional[pd.DataFrame] = None,
                            y_test: Optional[pd.Series] = None) -> Dict[str, float]:
             """
             Evaluates the model's performance using appropriate metrics.

             Args:
                 model (object, optional): Trained model to evaluate.
                 X_test (pd.DataFrame, optional): Test features.
                 y_test (pd.Series, optional): Test targets.

             Returns:
                 Dict: Dictionary of evaluation metrics.
             """
             model = model if model is not None else self.model
             X_test = X_test if X_test is not None else self.X_test
             y_test = y_test if y_test is not None else self.y_test

             if model is None:
                 raise ValueError("No trained model available. Call train_model() first or provide a model.")
             if X_test is None or y_test is None:
                 raise ValueError("Test data not provided. Call split_data() first or provide X_test and y_test.")

             logger.info("Evaluating model performance")

             if self.model_type == 'classifier':
                 y_pred = model.predict(X_test)
                 y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

                 metrics = {
                     'accuracy': accuracy_score(y_test, y_pred),
                     'precision': precision_score(y_test, y_pred, average='weighted'),
                     'recall': recall_score(y_test, y_pred, average='weighted'),
                     'f1_score': f1_score(y_test, y_pred, average='weighted'),
                 }

                 if y_pred_proba is not None:
                     metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                     metrics['log_loss'] = log_loss(y_test, y_pred_proba)

                 logger.info("\nClassification Report:")
                 logger.info(classification_report(y_test, y_pred))

                 try:
                     cm = confusion_matrix(y_test, y_pred)
                     plt.figure(figsize=(10, 8))
                     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                     plt.xlabel('Predicted labels')
                     plt.ylabel('True labels')
                     plt.title('Confusion Matrix')
                     if 'ipykernel' in sys.modules:
                         plt.show()
                     else:
                         plt.savefig(os.path.join(self.models_dir, 'confusion_matrix.png'))
                     plt.close()
                 except Exception as e:
                     logger.warning(f"Could not plot confusion matrix: {e}")

             else:
                 y_pred = model.predict(X_test)
                 metrics = {
                     'mean_squared_error': mean_squared_error(y_test, y_pred),
                     'root_mean_squared_error': np.sqrt(mean_squared_error(y_test, y_pred)),
                     'mean_absolute_error': mean_absolute_error(y_test, y_pred),
                     'r2_score': r2_score(y_test, y_pred),
                     'explained_variance': explained_variance_score(y_test, y_pred)
                 }

                 try:
                     plt.figure(figsize=(10, 6))
                     plt.scatter(y_test, y_pred, alpha=0.5)
                     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                     plt.xlabel('Actual values')
                     plt.ylabel('Predicted values')
                     plt.title('Actual vs Predicted Values')
                     if 'ipykernel' in sys.modules:
                         plt.show()
                     else:
                         plt.savefig(os.path.join(self.models_dir, 'actual_vs_predicted.png'))
                     plt.close()
                 except Exception as e:
                     logger.warning(f"Could not plot actual vs predicted values: {e}")

             logger.info("Evaluation metrics:")
             for metric_name, metric_value in metrics.items():
                 logger.info(f"{metric_name}: {metric_value:.4f}")

             self.metrics = metrics
             return metrics

