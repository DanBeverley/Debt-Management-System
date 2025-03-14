"""
Model Training Module for AI Debt Management System

Handles the training, evaluation, and persistence of machine learning models for debt management prediction tasks.
"""
import os
import sys
import time
import json
import logging
from typing import Optional, Dict, Tuple, Union, Optional, Any, Callable
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                                    StackingClassifier, RandomForestRegressor, StackingRegressor)
from sklearn.linear_model import  LogisticRegression, LinearRegression
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAPE = False

import joblib
import pickle
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

            test_size = test_size if test_size is not None else self.test_size
            val_size = val_size if val_size is not None else self.val_size
            # Separate features and target
            X = data.drop(columns = [target_col])
            y = data[target_col]
            # Determine if we should use stratification
            stratify_param = y if stratify and self.model_type == "classifier" else None
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = test_size, random_state = self.random_state,
                                                                                            stratify = stratify_param)
            # Second split: Separate validation set from training set
            if val_size > 0:
                val_size_adjusted = val_size / (1 - test_size)
                stratify_param_val = y_temp if stratify and self.model_type == "classifier" else None
                X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = val_size_adjusted,
                                                                                              random_state = self.random_state,
                                                                                              stratify = stratify_param_val)
                logger.info(f"Data split: Train={X_train.shape}, Validation={X_val.shape}, Test={X_test.shape}")
                # Save the split for later use
                self.X_train, self.y_train = X_train, y_train
                self.X_test, self.y_test = X_test, y_test
                self.X_val, self.y_val = X_val, y_val
                return X_train, y_train, X_test, y_test, X_val, y_val
            else:
                logger.info(f"Data split (no validation): Train={X_temp.shape}, Test={X_test.shape}")
                # Save the splits for later use
                self.X_train, self.y_train = X_temp, y_temp
                self.X_test, self.y_test = X_test, y_test
                self.X_val, self.y_val = None, None
                return X_temp, y_temp, X_test, y_test, None, None

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
                          learning_rate=0.05, iterations=300, depth=5, verbose=0,
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
