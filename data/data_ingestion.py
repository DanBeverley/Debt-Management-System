"""
Data Ingestion Module

This module handles loading data from various sources including
the Lending Club dataset, user inputs, databases, and APIs.
It includes optimizations for memory efficiency, validation,
and error handling.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Union, Optional, List, Tuple
import warnings
from pathlib import Path
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from sqlalchemy import create_engine
import requests
from tqdm import tqdm

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                    handlers = [logging.FileHandler('data_ingestion.log'),
                                                        logging.StreamHandler])
logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Class for handling data ingestion operations for the debt management system
    """
    def __init__(self, config:Optional[Dict] = None) -> None:
        """
        :param config: Configuration parameters for data ingestion
        """
        self.config = config or {}
        self.cache_dir  = self.config.get("cache_dir", "data/cache")
        self.sample_size = self.config.get("sample_size", 10000)
        self.use_dask = self.config.get("use_dask", False)
        self.use_parquet = self.config.get('use_parquet', False)
        self.chunk_size = self.config.get("chunk_size", 10000)
        # Cache directory if one has not exist
        os.makedirs(self.cache_dir, exist_ok = True)

    def load_lending_club_data(self,  file_path:str,
                                                    use_cache:bool =  True,
                                                    low_memory:bool = True,
                                                    columns:Optional[List[str]] = None) -> pd.DataFrame:
        """
        Loading the Lending Club dataset from csv

        :param file_path:           Path to the dataset
        :param use_cache:         Whether to use cache if available
        :param low_memory:    Whether to use memory-efficient loading
        :param columns:            Specific columns to load
        :return:                           Dataframe containing the Lending Club data
        """
        logger.info(f"Loading Lending Club data from {file_path}")
        file_hash = str(hash(f"{file_path}_{str(columns)}"))
        cache_file = os.path.join(self.cache_dir, f"lending_club_{file_hash}.parquet")

        if use_cache and os.path.exists(cache_file):
            logger.info(f"Loading from cache: {cache_file}")
            return pd.read_parquet(cache_file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist")

        file_ext = os.path.splitext(file_path)[1].lower()
        try:
            if self.use_dask:
                logger.info("Using Dask for memory-efficient loading")
                if file_ext == ".csv":
                    ddf = dd.read_csv(file_path, assume_missing = True,
                                                   usecols = columns, dtype_backend = "pyarrow")
                elif file_ext == ".parquet":
                    ddf = dd.read_parquet(file_path, columns = columns)
                else:
                    raise ValueError(f"Unsupported file format: {file_ext}")
                df = ddf.compute()
            else:
                   # Pandas with chunking for larger CSV files
                   if file_ext == ".csv":
                       if os.path.getsize(file_path) > 1_000_000_000: # !GB
                           logger.info(f"Larger file detected ({file_path}), using chunked reading")
                           chunks = []
                           with pd.read_csv(file_path,
                                                         chunksize = self.chunk_size,
                                                         usecols = columns,
                                                         low_memory = low_memory) as reader:
                               for chunk in tqdm(reader, desc = "Loading data chunks"):
                                    chunks.append(chunk)
                           df = pd.concat(chunks, ignore_index=True)
                       else:
                           df = pd.read_csv(file_path, usecols = columns, low_memory = low_memory)
                   elif file_ext == ".parquet":
                       df = pd.read_parquet(file_path, columns = columns)
                   elif file_ext == ".xlsx" or file_ext == ".xls":
                       if columns is not None:
                           df = pd.read_excel(file_path)
                           df = df[columns]
                       else:
                           df = pd.read_excel(file_path)
                   else:
                       raise ValueError(f"Unsupported file format: {file_ext}")
            if df.empty:
                raise ValueError("The loaded dataset is empty")
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

            # Cache the data if requested
            if use_cache:
                logger.info(f"Caching data to {cache_file}")
                df.to_parquet(cache_file, index = False)
            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def load_user_data(self, user_input: Union[Dict, str]) -> dict[str, pd.DataFrame]:
        """
        Parse and load financial data provided by the user

        :param user_input: User-provided financial data (e.g. , income, debts)
        :return: pd.DataFrame: Structured DataFrame containing user data
        """
        logger.info("Processing user input data")
        # Handle string input (JSON)
        if isinstance(user_input, str):
            try:
                user_input = json.loads(user_input)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON input: {e}")
                raise ValueError(f"Invalid JSON format: {e}")

        required_fields = ["income", "debts"]
        missing_fields = [field for field in required_fields if field not in user_input]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        # Process debts data
        if isinstance(user_input["debts"], list):
            debts_df = pd.DataFrame(user_input["debts"])

            required_debt_cols = ['amount', 'interest_rate', 'minimum_payment']
            missing_cols = [col for col in required_debt_cols if col not in debts_df.columns]
            if missing_cols:
                raise ValueError(f"Debt entries missing required fields: {', '.join(missing_cols)}")
        else:
            raise ValueError("'debts' must be a list of a debt items")

        user_profile = {"income": user_input.get('income'),
                                  "expenses": user_input.get('expenses', 0),
                                  "credit_score": user_input.get('credit_score'),
                                  "savings": user_input.get('savings', 0),
                                  "available_for_debt": user_input.get('available_for_debt'),
                                  "timestamp": datetime.now()}
        if  'available_for_debt' not in user_input:
            user_profile['available_for_debt'] = max(0, user_profile["income"] - user_profile["expenses"])
        user_df = pd.DataFrame([user_profile])
        logger.info(f"Processed user data with {len(debts_df)} debt entries")

        return {"user_profile": user_df,
                     "debts":debts_df}

    def load_from_database(self, connection_string:str, query:str) -> pd.DataFrame:
        """
        Load data from a database using SQL query

        :param connection_string:  Database connection string
        :param query:                      SQL query to execute
        :return:                                 DataFrame containing query results
        """
        logger.info(f"Loading data from database with query: {query[:100]}...")
        try:
            engine = create_engine(connection_string)
            df = pd.read_sql(query, engine)
            logger.info(f"Loaded {len(df)} rows from database")
            return df
        except Exception as e:
            logger.error(f"Database loading error: {e}")
            raise

    def load_from_api(self, api_url:str, params:Dict = None, headers:Dict = None) -> pd.DataFrame:
        """
        Load data from API endpoint

        :param api_url:      URL of the API endpoint
        :param params:     query parameters
        :param headers:    HTTP readers
        :return:                   DataFrame containing API response data
        """
        logger.info(f"Fetching data from API: {api_url}")
        try:
            response = requests.get(api_url, params = params, headers = headers)
            response.raise_for_status()

            data = response.json()
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and "results" in data:
                df = pd.DataFrame(data["results"])
            else:
                df = pd.DataFrame([data])
            logger.info(f"Loaded {len(df)} rows from API")
            return df
        except Exception as e:
            logger.error(f"API loading error: {e}")
            raise

    def validate_dataset(self, df:pd.DataFrame, rules:Dict = None) -> Dict:
            """
            Validate the dataset against a set of rule and return validation results

            :param df:         DataFrame to validate
            :param rule:      Validation rules
            :return:              Validation results containing issues found
            """
            logger.info("Validating dataset")
            results = {"missing_values": {},
                             "duplicates": 0,
                             "outliers": {},
                             "invalid_values": {},
                             "is_valid": True}
            # Check for missing values
            missing = df.isnull().sum()
            results["missing_values"] = missing[missing > 0].to_dict()
            duplicates = df.duplicated().sum()
            results["duplicates"] = duplicates
            # Apply custom validation rules
            if rules:
                for column, rule_dict in rules.items():
                    if column not in df.columns:
                        continue
                    # Range violations
                    if "min" in rule_dict and "max" in rule_dict:
                        min_val, max_val = rule_dict["min"] , rule_dict["max"]
                        mask = (df[column] < min_val) | (df[column] > max_val)
                        if mask.sum() > 0:
                            results["invalid_values"][column] = mask.sum()
                    # Invalid categories
                    if "categories" in rule_dict:
                        valid_cats = set(rule_dict["categories"])
                        invalid_count = (~df[column].isin(valid_cats)).sum()
                        if invalid_count > 0:
                            results["invalid_values"][column] = invalid_count
            if (sum(results["missing_values"].values()) > 0 or
                results["duplicates"] > 0 or
                len(results["invalid_values"]) > 0):
                results["is_valid"] = False
            logger.info(f"Validation complete: {'valid' if results['is_valid'] else 'invalid'} dataset")
            return results

    def clean_dataset(self, df:pd.DataFrame, strategies:Dict = None) -> pd.DataFrame:
           """
           Clean the dataset based on specified strategies

           :param df:                   DataFrame to clean
           :param strategies:      Cleaning strategies for columns
           :return:                        Cleaned DataFrame
           """
           logger.info("Cleaning dataset")
           df_clean = df.copy()
           default_strategies = {"missing_values":"drop",      # Options: drop, mean, median, mode, constant
                                              "duplicates":"drop",              # Options: drop, keep
                                              "outliers":"none"}                  # Options: none , clip, drop
           strategies = strategies or {}
           for key, value in default_strategies.items():
               strategies[key] = strategies.get(key, value)
           # Handle duplicates
           if strategies["duplicates"] == "drop":
               initial_count = len(df_clean)
               df_clean = df_clean.drop_duplicates()
               dropped_count = initial_count - len(df_clean)
               logger.info(f"Dropped {dropped_count} duplicate rows")
           # Handle missing values globally
           if strategies["missing_values"] == "drop":
               initial_count = len(df_clean)
               df_clean = df_clean.dropna()
               dropped_count = initial_count - len(df_clean)
               logger.info(f"Dropped {dropped_count} rows with missing values")
           # Handle column-specific strategies
           for column, strategy in strategies.get("columns", {}).items():
               if column not in df_clean.columns:
                   continue
              # Handle missing values
               if "missing" in strategy:
                   missing_strategy = strategy["missing"]
                   if missing_strategy == "mean" and pd.api.types.is_numeric_dtype(df_clean[column]):
                       df_clean[column] = df_clean[column].fillna(df_clean[column].mean())
                   elif missing_strategy == "median" and pd.api.types.is_numeric_dtype(df_clean[column]):
                       df_clean[column] = df_clean[column].fillna(df_clean[column].median())
                   elif missing_strategy == "mode":
                       df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])
                   elif missing_strategy == "constant" and "value" in strategy:
                       df_clean[column] = df_clean[column].fillna(strategy["value"])
              # Handle outliers for numeric columns
               if "outliers" in strategy and pd.api.types.is_numeric_dtype(df_clean[column]):
                   outlier_strategy = strategy["outliers"]
                   if outlier_strategy == "clip":
                       q1 = df_clean[column].quantile(0.25)
                       q3 = df_clean[column].quantile(0.75)
                       iqr = q3 - q1
                       lower_bound = q1 - 1.5 * iqr
                       upper_bound = q3 + 1.5 * iqr
                       df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)
           logger.info(f"Cleaning complete. Rows remaining: {len(df_clean)}")
           return df_clean

    def sample_data(self, df:pd.DataFrame, n:int = None,
                                 frac:float = None, stratify_column:str = None) -> pd.DataFrame:
           """
           Create a sample of the dataset for development or testing

           :param df:                         DataFrame to sample
           :param n:                           Number of samples to return
           :param frac:                      Fraction of data to sample
           :param stratify_column:   Column to use for stratified sampling
           :return:                              Sampled DataFrame
           """
           if n is None and frac is None:
               n = min(self.sample_size, len(df))
           logger.info(f"Creating: {'stratified' if stratify_column else 'random'} sample of data")
           if stratify_column and stratify_column in df.columns:
               # Stratified sampling
               try:
                   if n is not None:
                       from sklearn.model_selection import train_test_split
                       _, sample = train_test_split(df, test_size=n / len(df),
                                                                     stratify = df[stratify_column],
                                                                     random_state = 42)
                       return sample
                   else:
                       return df.groupby(stratify_column, group_keys=False).apply(lambda x: x.sample(frac = frac, random_state = 42))
               except Exception as e:
                   logger.warning(f"Stratified sampling failed: {e} . Falling back to random sampling")
                   stratify_column = None
           if n is not None:
               return df.sample(n = min(n, len(df)), random_state = 42)
           else:
               return df.sample(frac = frac, random_state = 42)

    def save_processed_data(self, df:pd.DataFrame, output_path:str, format:str = "parquet", **kwargs) -> None:
           """
           Save processed data to disk

           :param df:                      DataFrame to save
           :param output_path:     Path where to save the data
           :param format:               File format ('csv', 'parquet', 'excel')
           :param kwargs:               Additional arguments for the save function
           :return:                            None
           """
           logger.info(f"Saving processed data to {output_path} in {format} format")
           # Create directory if has not exist
           os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok = True)
           try:
               if format.lower() == "cpu":
                   df.to_csv(output_path, index = False, **kwargs)
               elif format.lower() == "parquet":
                   df.to_parquet(output_path, index = False, **kwargs)
               elif format.lower() == "excel":
                   df.to_excel(output_path, index = False, **kwargs)
               else:
                   raise ValueError(f"Unsupported format: {format}")
               logger.info(f"Data succesfully saved to {output_path}")
           except Exception as e:
               logger.error(f"Failed to save data: {e}")
               raise

    def generate_data_profile(self, df: pd.DataFrame, output_path:Optional[str] = None) -> Dict:
          """
          Generate a profile report of the dataset
          :param df:                     DataFrame to profile
          :param output_path:    Path to save HTML
          :return:                          Dictionary containing profile statistics
          """
          logger.info("Generating data profile")
          profile = {
              'rows': len(df),
              'columns': len(df.columns),
              'dtypes': df.dtypes.astype(str).to_dict(),
              'missing_values': df.isnull().sum().to_dict(),
              'missing_percentage': (df.isnull().mean() * 100).round(2).to_dict(),
              'summary': {}}
          # Generate summary statistics
          for column in df.columns:
              if pd.api.types.is_numeric_dtype(df[column]):
                  profile["summary"][column] = {
                    'mean': float(df[column].mean()) if not pd.isna(df[column].mean()) else None,
                    'median': float(df[column].median()) if not pd.isna(df[column].median()) else None,
                    'std': float(df[column].std()) if not pd.isna(df[column].std()) else None,
                    'min': float(df[column].min()) if not pd.isna(df[column].min()) else None,
                    'max': float(df[column].max()) if not pd.isna(df[column].max()) else None
                }
              elif pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
                  value_counts = df[column].value_counts()
                  profile["summary"][column] = {
                                                "unique_values": len(value_counts),
                                                "top_values":value_counts.head(5).to_dict() if not value_counts.empty else {}}
          if output_path:
              try:
                  from pandas_profiling import ProfileReport
                  prof = ProfileReport(df, tilte = "Lending Club Data profile", minimal = True, explorative = True)
                  prof.to_file(output_path)
                  logger.info(f"HTML profile report saved to {output_path}")
              except ImportError:
                  logger.warning("pandas-profiling not installed. HTML report not generated")
          return profile
