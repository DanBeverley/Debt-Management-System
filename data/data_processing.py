"""
Data processing Module for AI Debt Management System

The module handles cleaning, transformation, and preparation of data for machine learning models in the
debt management system. Includes functions for handling missing values, encoding categorical variables,
normalizing numerical features, and feature engineering specific to lending and debt management
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from sklearn.preprocessing import (
StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder,
OrdinalEncoder)
import category_encoders as ce
import os
from scipy import stats

logging.basicConfig(level = logging.INFO,
                    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers = [logging.FileHandler('data_processing.log'),
                                        logging.StreamHandler()])
logger = logging.getLogger(__name__)

class DataPreprocessor:
         def __init__(self, config: Optional[Dict] = None) -> None:
               self.config = config or {}
               expected_keys = {
                   "model_dir":str,
                   "categorical_threshold":int,
                   "outlier_threshold":float,
                   "scaling_method":str,
                   "encoding_method":str,
                   "imputation_method":str,
                   "variance_threshold":float,
                   "feature_importance_method":str
               }

               for key, value in self.config.items():
                   if key not in expected_keys:
                       logger.warning(f"Unexpected config key: {key}")
                   elif not isinstance(value, expected_keys.get(key, object)):
                       raise ValueError(f"Invalid type for {key}: expected {expected_keys[key]}, got {type(value)}")

               self.models_dir = self.config.get("models_dir", "models")
               self.categorical_threshold = self.config.get("categorical_threshold", 10)
               self.outlier_threshold = self.config.get("outlier_threshold", 3.0)
               self.scaling_method = self.config.get("scaling_method", "standard")
               self.encoding_method = self.config.get("encoding_method", "onehot")
               self.imputation_method = self.config.get("imputation_method", "mean")
               self.variance_threshold = self.config.get("variance_threshold", 0.01)
               self.feature_importance_method = self.config.get("feature_importance_method", "mutual_info")

               if not os.path.exists(self.models_dir):
                   os.makedirs(self.models_dir, exist_ok = True)

               self.scalers = {}
               self.encoders = {}
               self.imputers = {}
               self.feature_selectors = {}

         def clean_data(self, data:pd.DataFrame, target_column:Optional[str]=None,
                       drop_columns:Optional[List[str]] = None, missing_threshold:float = 0.5,
                       outlier_thresholds:Optional[Dict[str, float]] = None) -> pd.DataFrame:
            """
            Clean the dataset by handling missing values, duplicates and outliers

            :param data:                     Raw input data
            :param target_column:    Name of the target column to preserve
            :param drop_columns:     Columns to drop from the dataset
            :param missing_threshold:  Threshold for dropping columns with missing values (0 to 1)
            :param outlier_thresholds:   Custom outliers thresholds per column
            :return:                              Cleaned DataFrame
            """
            logger.info("Starting data cleaning process")
            df = data.copy()
            initial_shape = df.shape
            logger.info(f"Initial Shape: {initial_shape}")
            # Drop specified columns
            if drop_columns:
                df = df.drop(columns = [col for col in drop_columns if col in df.columns])
                logger.info(f"Dropped  columns: {[col for col in drop_columns if col in data.columns]}")
            # Handle duplicates
            if df.duplicated().sum() > 0:
                df = df.drop_duplicates()
                logger.info(f"Removed {df.duplicated().sum()} duplicate rows")
            # Type conversions and fixing
            for col in df.columns:
                  # Convert incorrectly parsed numeric columns
                  if df[col].dtypes == "object":
                      try:
                            # Try to convert string percentage to float (e.g, "5.2%" -> 0.052)
                            if df[col].str.contains("%").any():
                                df[col] = df[col].str.rstrip("%").astype('float') / 100
                                logger.info(f"Converted percentage column '{col}' to float")
                            # Try to convert currency to float (e.g. "$1,234.56" -> 1234.56)
                            elif df[col].str.contains('[$,]').any():
                                df[col] = df[col].replace("[\$,]", " ", regex = True).astype("float")
                                logger.info(f"Converted currency column '{col}' to float")
                            # Try standard numeric conversion
                            else:
                                  df[col] = pd.to_numeric(df[col], errors = "ignore")
                      except Exception as e:
                            # Keep as object if conversion fails
                            logger.warning(f"Could not convert column '{col}': {e}")
            # Drop columns with more than missing_threshold missing values
            missing_pct = df.isnull().mean()
            high_missing_cols = missing_pct[missing_pct > missing_threshold].index.tolist()
            if high_missing_cols:
                if target_column in high_missing_cols:
                    high_missing_cols.remove(target_column)
                df = df.drop(columns = high_missing_cols)
                logger.info(f"Dropped columns with >{missing_threshold:.0%} missing values: {high_missing_cols}")
            # Fill Missing values based on data type
            numeric_cols = df.select_dtypes(include=["number"]).columns
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns
            # Impute missing numeric values with median
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    # If it's target column and has missing values, we drop those rows
                    if col == target_column:
                        df = df.dropna(subset = [col])
                        logger.info(f"Dropped {df[col].isnull().sum()} rows with missing target values")
                    else:
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val)
                        logger.info(f"Filled {df[col].isnull().sum()} missing values in '{col}' with median ({median_val})")
            # Impute missing categorical values with most frequent
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    # If it's target column with missing values, drop those rows
                    if col == target_column:
                        df = df.dropna(subset = [col])
                        logger.info(f"Dropped {df[col].isnull().sum()} rows with missing target values")
                    else:
                        mode_val = df[col].mode().iloc[0]
                        df[col] = df[col].fillna(mode_val)
                        logger.info(f"Filled {df[col].isnull().sum()} missing values in '{col}' with mode ({mode_val})")
            # Handle outliers in numeric columns
            for col in numeric_cols:
                if col == target_column:
                    continue  # Don't adjust the target column
                # Get outlier threshold (default IQR method)
                threshold = outlier_thresholds.get(col, 1.5) if outlier_thresholds else 1.5
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                # Define upper and lower bounds
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                # Count outliers
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    logger.info(f"Found {outliers} outliers in '{col}'")
                    # Clip outliers to the boundaries
                    df[col] = df[col].clip(lower = lower_bound, upper = upper_bound)
                    logger.info(f"Clipped {outliers} outliers in '{col}' to [{lower_bound:.2f}, {upper_bound:.2f}]")
            logger.info(f"Final Shape after cleaning: {df.shape}")
            return df

         def encode_categorical(self, data:pd.DataFrame, columns:Optional[List[str]] = None,
                                                 target_col: Optional[str] = None, training:bool = True,
                                                 handle_missing:str = "error") -> pd.DataFrame:
             """
             Convert categorical variables into numerical presentations
             :param data:                      Input data
             :param columns:               Specific columns to encode (if None, auto-detect)
             :param target_col:             Target column name for target encoding
             :param training:                Training or predicting data
             :param handle_missing:    Ways to handle missing values ('error', 'placeholder', 'mode')
             :return:                               DataFrame with encoded categorical columns
             """
             logger.info("Starting categorical encoding")
             df = data.copy()

             if columns is None:
                 categorical_cols = list(df.select_dtypes(include=["category", "object"]).columns)
                 for col in df.select_dtypes(include = ["numbers"]).columns:
                     if df[col].nunique() <= self.categorical_threshold:
                         categorical_cols.append(col)
                 if target_col and target_col in categorical_cols:
                     categorical_cols.remove(target_col)
             else:
                categorical_cols = [col for col in columns if col in df.columns]
             if not categorical_cols:
                 logger.info("No categorical columns found for encoding")
                 return df

             logger.info(f"Encoding {len(categorical_cols)} categorical columns:  {categorical_cols}")

             for col in categorical_cols:
                  if df[col].isnull().any():
                      if handle_missing == "error":
                          raise ValueError(f"Column '{col}' contains missing values")
                      elif handle_missing == "placeholder":
                          df[col] = df[col].fillna("__MISSING__")
                      elif handle_missing == "mode":
                          df[col] = df[col].fillna(df[col].mode()[0])

                  if training:
                      if self.encoding_method == "onehot":
                          encoder = OneHotEncoder(drop = "first", sparse_output=False)
                          encoded_data = encoder.fit_transform(df[[col]])
                          encoded_df = pd.DataFrame(encoded_data, columns =[f"{col}_{cat}" for cat in encoder.categories_[0][1:]],
                                                    index = df.index)
                          df = pd.concat([df.drop(columns = [col]), encoded_df], axis = 1)
                      elif self.encoding_method == "label":
                          le = LabelEncoder()
                          df[col] = le.fit_transform(df[col].astype(str))
                          self.encoders[col] = le
                      elif self.encoding_method == "target" and target_col:
                          te = ce.TargetEncoder(cols = [col])
                          df[col] = te.fit_transform(df[col], df[target_col])
                          self.encoders[col] = te
                      elif self.encoding_method == "binary":
                          be = ce.BinaryEncoder(cols = [col])
                          df = be.fit_transform(df)
                          self.encoders[col] = be
                      elif self.encoding_method == "frequency":
                          freq = df[col].value_counts(normalize = True)
                          df[col] = df[col].map(freq)
                          self.encoders[col] = freq.to_dict()
                      else:
                          oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value = -1)
                          df[col] = oe.fit_transform(df[[col]])
                          self.encoders[col] = oe
                      logger.info(f"Created and saved encoder for column '{col}' using '{self.encoding_method}'")
                  else:
                      if col in self.encoders:
                          encoder = self.encoders[col]
                          if self.encoding_method == 'onehot':
                              df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
                          elif self.encoding_method == 'label':
                              df[col] = encoder.transform(df[col].astype(str))
                          elif self.encoding_method == 'target':
                              df[col] = encoder.transform(df[col])
                          elif self.encoding_method == 'binary':
                              df = encoder.transform(df)
                          elif self.encoding_method == 'frequency':
                              df[col] = df[col].map(encoder)
                          else:
                              df[col] = encoder.transform(df[[col]])
                      else:
                        logger.warning(f"No encoder found for column '{col}' during prediction")

             logger.info("Categorical encoding completed")
             return df

         def normalize_data(self, data:pd.DataFrame, columns: Optional[List[str]] = None,
                                          training:bool = True) -> pd.DataFrame:
              """
              Scale numerical features to a standard range

              :param data:             Data with numerical columns
              :param columns:      Specific columns to scale (if None, auto-detect)
              :param training:       Training or prediction data
              :return:                      DataFrame with scaled numerical columns
              """
              logger.info("Starting data normalization")
              df = data.copy()

              if columns is None:
                  numerical_cols = list(df.select_dtypes(include = ["number"]).columns)
              else:
                  numerical_cols = [col for col in columns if col in df.columns]
              if not numerical_cols:
                  logger.info("No numerical columns for normalization")
                  return df
              logger.info(f"Normalizing {len(numerical_cols)} numerical columns: {numerical_cols}")
              for col in numerical_cols:
                  if training:
                      if self.scaling_method == "standard":
                          scaler = StandardScaler()
                      elif self.scaling_method == "minmax":
                          scaler = MinMaxScaler()
                      elif self.scaling_method == "robust":
                          scaler = RobustScaler()
                      else:
                          scaler = StandardScaler()
                      df[col] = scaler.fit_transform(df[[col]])
                      self.scalers[col] = scaler
                  else:
                      if col in self.scalers:
                          scaler = self.scalers[col]
                          df[col] = scaler.transforms(df[[col]])
                      else:
                          logger.warning(f"No scaler found for column '{col}' during prediction")
              logger.info("Data normalization completed")
              return df

         @staticmethod
         def feature_engineering(data:pd.DataFrame) -> pd.DataFrame:
             """
             Create new features from existing data

             :param data:  Input data
             :return:           DataFrame with new features
             """
             logger.info("Starting feature engineering")
             df = data.copy()
             # Debt-to-income ratio
             if 'total_debt' in df.columns and 'income' in df.columns:
                 df['debt_to_income'] = df['total_debt'] / df['income'].replace(0, np.nan)
                 logger.info("Added debt-to-income ratio feature")
            # Credit utilization
             if 'credit_limit' in df.columns and 'credit_used' in df.columns:
                 df['credit_utilization'] = df['credit_used'] / df['credit_limit'].replace(0, np.nan)
                 logger.info("Added credit utilization feature")
            # Age from birthdate
             if "birth_date" in df.columns:
                 df["age"] = (pd.to_datetime("today") - pd.to_datetime(df["birth_date"], errors = "coerce")).dt.days / 365.25
                 logger.info("Added age feature from birth_date")
             logger.info("Feature engineering completed")
             return df






