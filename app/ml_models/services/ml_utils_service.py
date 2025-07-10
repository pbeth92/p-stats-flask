import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import logging

from app.dataset.services.dataset_service import DatasetService
from app.dataset.services.dataset_records_service import DatasetRecordsService

class MLUtilsService:
    def __init__(self):
        self.dataset_service = DatasetService()
        self.dataset_records_service = DatasetRecordsService()

    def train_test_split(self, data):
        """
        Divide el dataset en conjuntos de entrenamiento y prueba.
        
        Parameters:
            data (dict): Diccionario que incluye 'dataset_id', 'test_size', 'predictors', y 'target'.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        try:
            # Get required variables for model
            variables = data["predictors"] + [data["target"]]
            
            # Get records with only required variables
            dataset = self.dataset_records_service.get_records_by_variables(
                str(data["dataset_id"]),
                variables
            )
            df = pd.DataFrame([record.to_mongo() for record in dataset['records']])

            # Validate predictors exist in dataset
            missing_predictors = [col for col in data["predictors"] if col not in df.columns]
            if missing_predictors:
                raise ValueError(f"The following predictor columns are missing in the dataset: {missing_predictors}")
        
            # Validate target exists in dataset
            if data["target"] not in df.columns:
                raise ValueError(f"Target column '{data['target']}' is missing in the dataset.")
            
            X = df[data["predictors"]]
            y = df[data["target"]]

            # Use sklearn's train_test_split with default random_state if not provided
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=data["test_size"], random_state=data.get("random_state", 42)
            )

            return X_train, X_test, y_train, y_test

        except KeyError as ke:
            # Error específico si falta una clave en el diccionario de entrada
            raise ValueError(f"Missing key in data input: {ke}")
        except Exception as e:
            logging.error(f"An error occurred during train-test split: {e}")
            raise ValueError("Failed to split dataset into training and test sets.")

    def evaluate_regression_metrics(self, y_true, y_pred, model=None, X=None, cv_folds=None):
        """
        Calculate regression metrics
        
        Parameters:
            y_true: Actual target values
            y_pred: Predicted target values
            model: Optional - Fitted model for cross-validation
            X: Optional - Feature matrix for cross-validation
            cv_folds: Optional - Number of cross-validation folds
            
        Returns:
            dict: Dictionary containing regression metrics
        """
        try:
            metrics = {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'r2': float(r2_score(y_true, y_pred))
            }
            
            # Add cross-validation score if model and data are provided
            if model is not None and X is not None and cv_folds is not None:
                cv_scores = cross_val_score(model, X, y_true, cv=cv_folds, scoring='r2')
                metrics['cv_r2_mean'] = float(cv_scores.mean())
                metrics['cv_r2_std'] = float(cv_scores.std())
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating regression metrics: {e}")
            raise ValueError(f"Failed to calculate regression metrics: {str(e)}")

    def prepare_prediction_data(self, feature_data, feature_variables):
        """
        Prepare feature data for making predictions
        
        Parameters:
            feature_data: Dictionary or list of dictionaries containing feature values
            feature_variables: List of feature variable names expected by the model
            
        Returns:
            numpy.ndarray: Prepared feature matrix for prediction
        """
        try:
            # Convert to DataFrame if it's a dictionary or list
            if isinstance(feature_data, dict):
                # Single prediction - convert to list of one dictionary
                feature_data = [feature_data]
            elif isinstance(feature_data, list):
                # Multiple predictions
                pass
            else:
                raise ValueError("feature_data must be a dictionary or list of dictionaries")
            
            # Create DataFrame
            df = pd.DataFrame(feature_data)
            
            # Validate that all required features are present
            missing_features = [f for f in feature_variables if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Select only the required features in the correct order
            X = df[feature_variables].values
            
            return X
            
        except Exception as e:
            logging.error(f"Error preparing prediction data: {e}")
            raise ValueError(f"Failed to prepare prediction data: {str(e)}")