from sklearn.linear_model import LinearRegression
import numpy as np
import logging

from app.ml_models.services.ml_utils_service import MLUtilsService
from app.ml_models.models.ml_model import MLModel
from app.shared.shared_enums import ModelType
from app.dataset.models.dataset_model import Dataset

class MLRegressionService:
    def __init__(self):
        self.ml_utils = MLUtilsService()
        
    def train_linear_regression(self, dataset_id, config):
        """
        Train a linear regression model with the specified configuration.
        
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
            model = LinearRegression(**hyperparameters)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics using shared functionality
            train_metrics = self.ml_utils.evaluate_regression_metrics(
                y_train, y_pred_train, model, X_train,
                config.get("cross_validation_folds", 5)
            )
            test_metrics = self.ml_utils.evaluate_regression_metrics(
                y_test, y_pred_test
            )
            
            # Create coefficients dictionary
            coefficients = dict(zip(config["feature_variables"], 
                                 [str(coef) for coef in model.coef_]))
            
            # Create and save model in database
            ml_model = MLModel(
                name=config["name"],
                description=config.get("description", ""),
                dataset=dataset,
                model_type=ModelType.LINEAR_REGRESSION,
                feature_variables=config["feature_variables"],
                target_variable=config["target_variable"],
                hyperparameters=hyperparameters,
                coefficients=coefficients,
                intercept=str(model.intercept_),
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
            logging.error(f"Error training linear regression model: {e}")
            raise ValueError(f"Failed to train linear regression model: {str(e)}")
    
    def predict(self, model_id, feature_data):
        """
        Make predictions using a trained linear regression model.
        
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
            
            # Make predictions
            predictions = np.dot(X, coefficients) + intercept
            
            return {
                "predictions": predictions.tolist(),
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

    def get_linear_regression_plot_data(self, model_id):
        """
        Get data for plotting the regression line and scatter plot.
        
        Parameters:
            model_id: ID of the trained model to use
            
        Returns:
            dict: Data for plotting including:
                - original_points: Dictionary with x and y arrays for scatter plot
                - regression_line: Dictionary with x and y arrays for the regression line
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
            X = np.array([record[model.feature_variables[0]] for record in records])  # For simple linear regression
            y = np.array([record[model.target_variable] for record in records])
            
            # For the regression line, create points spanning the X range
            x_min = np.min(X)
            x_max = np.max(X)
            x_line = np.linspace(x_min, x_max, 100)
            
            # Calculate regression line points using coefficients and intercept
            coef = float(model.coefficients[model.feature_variables[0]])  # For simple linear regression
            intercept = float(model.intercept)
            y_line = coef * x_line + intercept
            
            return {
                "original_points": {
                    "x": X.tolist(),
                    "y": y.tolist()
                },
                "regression_line": {
                    "x": x_line.tolist(),
                    "y": y_line.tolist()
                }
            }
            
        except MLModel.DoesNotExist:
            error_msg = f"Model with id {model_id} not found"
            logging.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            logging.error(f"Error getting plot data: {e}")
            raise ValueError(f"Failed to get plot data: {str(e)}")
