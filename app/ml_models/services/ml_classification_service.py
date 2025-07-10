from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import logging

from app.ml_models.services.ml_utils_service import MLUtilsService
from app.ml_models.models.ml_model import MLModel
from app.shared.shared_enums import ModelType
from app.dataset.models.dataset_model import Dataset

class MLClassificationService:
    def __init__(self):
        self.ml_utils = MLUtilsService()
        
    def train_logistic_regression(self, dataset_id, config):
        """
        Train a logistic regression model with the specified configuration.
        
        Parameters:
            dataset_id: ID of the dataset to use
            config: Dictionary containing model configuration
                   (target_variable, feature_variables, test_size, etc.)
        
        Returns:
            dict: Model results including metrics and model info
        """
        try:
            # Get dataset
            dataset = Dataset.objects.get(id=dataset_id)
            
            # Prepare data for training
            data = {
                "dataset_id": dataset_id,
                "test_size": config["test_size"],
                "random_state": config["random_state"],
                "predictors": config["feature_variables"],
                "target": config["target_variable"]
            }
            
            # Use shared functionality for train-test split
            X_train, X_test, y_train, y_test = self.ml_utils.train_test_split(data)
            
            # Initialize and train model with hyperparameters
            hyperparameters = config.get("hyperparameters", {})
            model = LogisticRegression(**hyperparameters)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Get prediction probabilities for AUC calculation
            try:
                y_pred_train_proba = model.predict_proba(X_train)[:, 1]
                y_pred_test_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                # In case of multiclass or other issues
                y_pred_train_proba = None
                y_pred_test_proba = None
            
            # Calculate metrics
            train_metrics = self._evaluate_classification_metrics(
                y_train, y_pred_train, y_pred_train_proba, model, X_train,
                config.get("cross_validation_folds", 5)
            )
            test_metrics = self._evaluate_classification_metrics(
                y_test, y_pred_test, y_pred_test_proba
            )
            
            # Create coefficients dictionary
            coefficients = dict(zip(config["feature_variables"], 
                                 [str(coef) for coef in model.coef_[0]]))
            
            # Create and save model in database
            ml_model = MLModel(
                name=config["name"],
                description=config.get("description", ""),
                dataset=dataset,
                model_type=ModelType.LOGISTIC_REGRESSION,
                feature_variables=config["feature_variables"],
                target_variable=config["target_variable"],
                hyperparameters=hyperparameters,
                coefficients=coefficients,
                intercept=str(model.intercept_[0]),
                metrics={
                    "train": train_metrics,
                    "test": test_metrics
                }
            )
            ml_model.save()
            
            return ml_model.to_dict()
            
        except Dataset.DoesNotExist:
            error_msg = f"Dataset with id {dataset_id} not found"
            logging.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            logging.error(f"Error training logistic regression model: {e}")
            raise ValueError(f"Failed to train logistic regression model: {str(e)}")
    
    def predict(self, model_id, feature_data):
        """
        Make predictions using a trained logistic regression model.
        
        Parameters:
            model_id: ID of the trained model to use
            feature_data: Dictionary or DataFrame containing feature values
            
        Returns:
            dict: Prediction results
        """
        try:
            # Get model from database
            model = MLModel.objects.get(id=model_id)
            
            # Prepare prediction data using shared functionality
            X = self.ml_utils.prepare_prediction_data(
                feature_data,
                model.feature_variables
            )
            
            # Convert coefficients and intercept back to float
            coefficients = [float(model.coefficients[f]) for f in model.feature_variables]
            intercept = float(model.intercept)
            
            # Calculate logistic regression predictions manually
            # z = X * coefficients + intercept
            z = np.dot(X, coefficients) + intercept
            # Apply sigmoid function: p = 1 / (1 + e^(-z))
            probabilities = 1 / (1 + np.exp(-z))
            # Convert probabilities to binary predictions (threshold 0.5)
            predictions = (probabilities > 0.5).astype(int)
            
            return {
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist(),
                "model_name": model.name,
                "feature_variables": model.feature_variables
            }
            
        except MLModel.DoesNotExist:
            error_msg = f"Model with id {model_id} not found"
            logging.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            raise ValueError(f"Failed to make predictions: {str(e)}")

    def get_logistic_regression_plot_data(self, model_id):
        """
        Get data for plotting the logistic regression curve and scatter plot.
        
        Parameters:
            model_id: ID of the trained model to use
            
        Returns:
            dict: Data for plotting including:
                - original_points: Dictionary with x and y arrays for scatter plot
                - logistic_curve: Dictionary with x and y arrays for the logistic curve
        """
        try:
            from app.dataset.services.dataset_records_service import DatasetRecordsService
            records_service = DatasetRecordsService()

            # Get model from database
            model = MLModel.objects.get(id=model_id)
            
            # Get variables to fetch
            variables = model.feature_variables + [model.target_variable]
            
            # Get records from dataset
            result = records_service.get_records_by_variables(
                str(model.dataset.id),
                variables
            )
            
            # Extract X and y data from records
            records = list(result['records'])
            X = np.array([record[model.feature_variables[0]] for record in records])  # For simple logistic regression
            y = np.array([record[model.target_variable] for record in records])
            
            # For the logistic curve, create points spanning the X range
            x_min = np.min(X)
            x_max = np.max(X)
            x_curve = np.linspace(x_min, x_max, 100)
            
            # Calculate logistic curve points using coefficients and intercept
            coef = float(model.coefficients[model.feature_variables[0]])  # For simple logistic regression
            intercept = float(model.intercept)
            z = coef * x_curve + intercept
            y_curve = 1 / (1 + np.exp(-z))  # Sigmoid function
            
            return {
                "original_points": {
                    "x": X.tolist(),
                    "y": y.tolist()
                },
                "logistic_curve": {
                    "x": x_curve.tolist(),
                    "y": y_curve.tolist()
                }
            }
            
        except MLModel.DoesNotExist:
            error_msg = f"Model with id {model_id} not found"
            logging.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            logging.error(f"Error getting plot data: {e}")
            raise ValueError(f"Failed to get plot data: {str(e)}")

    def _evaluate_classification_metrics(self, y_true, y_pred, y_pred_proba=None, model=None, X=None, cv_folds=None):
        """
        Calculate classification metrics
        
        Parameters:
            y_true: Actual target values
            y_pred: Predicted target values
            y_pred_proba: Predicted probabilities (optional)
            model: Optional - Fitted model for cross-validation
            X: Optional - Feature matrix for cross-validation
            cv_folds: Optional - Number of cross-validation folds
            
        Returns:
            dict: Dictionary containing classification metrics
        """
        try:
            # Get unique classes
            unique_classes = np.unique(y_true)
            
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred))
            }
            
            # Calculate precision, recall, and F1 score
            # Use different averaging strategies based on number of classes
            if len(unique_classes) == 2:
                # Binary classification
                metrics.update({
                    'precision': float(precision_score(y_true, y_pred)),
                    'recall': float(recall_score(y_true, y_pred)),
                    'f1_score': float(f1_score(y_true, y_pred))
                })
                
                # Add AUC if probabilities are available
                if y_pred_proba is not None:
                    try:
                        metrics['auc_roc'] = float(roc_auc_score(y_true, y_pred_proba))
                    except Exception:
                        # Skip AUC if there's any issue
                        pass
                        
            else:
                # Multiclass classification
                metrics.update({
                    'precision_macro': float(precision_score(y_true, y_pred, average='macro')),
                    'recall_macro': float(recall_score(y_true, y_pred, average='macro')),
                    'f1_score_macro': float(f1_score(y_true, y_pred, average='macro')),
                    'precision_weighted': float(precision_score(y_true, y_pred, average='weighted')),
                    'recall_weighted': float(recall_score(y_true, y_pred, average='weighted')),
                    'f1_score_weighted': float(f1_score(y_true, y_pred, average='weighted'))
                })
            
            # Add confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Add cross-validation score if model and data are provided
            if model is not None and X is not None and cv_folds is not None:
                cv_scores = cross_val_score(model, X, y_true, cv=cv_folds, scoring='accuracy')
                metrics['cv_accuracy_mean'] = float(cv_scores.mean())
                metrics['cv_accuracy_std'] = float(cv_scores.std())
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating classification metrics: {e}")
            raise ValueError(f"Failed to calculate classification metrics: {str(e)}")