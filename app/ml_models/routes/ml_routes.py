from flask import Blueprint, jsonify, request
from marshmallow import ValidationError

from app.ml_models.schemas.linear_regression_schema import CreateLinearRegressionModelSchema
from app.ml_models.schemas.logistic_regression_schema import CreateLogisticRegressionModelSchema
from app.ml_models.services.ml_regression_service import MLRegressionService
from app.ml_models.services.ml_classification_service import MLClassificationService
from app.ml_models.models.ml_model import MLModel
from app.shared.shared_enums import ModelType

ml_routes = Blueprint('ml', __name__, url_prefix='/ml')
ml_regression_service = MLRegressionService()
ml_classification_service = MLClassificationService()


# ============================================================================
# GENERAL MODEL ROUTES - Model Management and Information
# ============================================================================

@ml_routes.route('/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """Get model information by ID"""
    try:
        model = MLModel.objects.get(id=model_id)
        
        return jsonify(model.to_dict()), 200
    except MLModel.DoesNotExist:
        return jsonify({"error": f"Model with id {model_id} not found"}), 404
    except Exception as err:
        return jsonify({"error": "Internal server error"}), 500

@ml_routes.route('/models', methods=['GET'])
def get_models():
    """Get all models filtered by model_type and/or dataset_id if specified"""
    try:
        model_type = request.args.get('model_type')
        dataset_id = request.args.get('dataset_id')
        filter_params = {}
        
        if model_type:
            try:
                model_type_enum = ModelType(model_type)
                filter_params['model_type'] = model_type_enum
            except ValueError:
                return jsonify({"error": f"Invalid model type. Valid types are: {[t.value for t in ModelType]}"}), 400
                
        if dataset_id:
            filter_params['dataset'] = dataset_id

        print(filter_params)
        
        models = MLModel.objects.filter(**filter_params)
        models_data = [model.to_dict() for model in models]
        return jsonify(models_data), 200
    except Exception as err:
        return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# LINEAR REGRESSION ROUTES - Regression Model Training and Prediction
# ============================================================================
    
@ml_routes.route('/linear-regression/<dataset_id>', methods=['POST'])
def linear_regression(dataset_id):
    """Create and train a linear regression model"""
    if request.method == 'POST':
        request_data = request.json
        schema = CreateLinearRegressionModelSchema()

        try:
            # Validate request data
            config = schema.load(request_data)
            
            # Train model and get results
            model_info = ml_regression_service.train_linear_regression(dataset_id, config)
            
            return jsonify(model_info), 201
        
        except ValidationError as err:
            return jsonify(err.messages), 400
        except ValueError as err:
            return jsonify({"error": str(err)}), 400
        except Exception as err:
            return jsonify({"error": "Internal server error"}), 500
    
@ml_routes.route('/linear-regression/<model_id>/plot-data', methods=['GET'])
def get_linear_regression_model_plot_data(model_id):
    """Get all linear regression models"""
    try:
        plot_data = ml_regression_service.get_linear_regression_plot_data(model_id)

        return jsonify(plot_data), 200
    except Exception as err:
        return jsonify({"error": "Internal server error"}), 500
    
@ml_routes.route('/linear-regression/predict/<model_id>', methods=['POST'])
def predict_linear_regression(model_id):
    """Make predictions using a trained linear regression model"""
    try:
        feature_data = request.json.get("feature_data")
        
        if not feature_data:
            return jsonify({"error": "Missing feature_data"}), 400
            
        predictions = ml_regression_service.predict(model_id, feature_data)
        return jsonify(predictions), 200
        
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception as err:
        return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# LOGISTIC REGRESSION ROUTES - Classification Model Training and Prediction
# ============================================================================

@ml_routes.route('/logistic-regression/<dataset_id>', methods=['POST'])
def logistic_regression(dataset_id):
    """Create and train a logistic regression model"""
    if request.method == 'POST':
        request_data = request.json
        schema = CreateLogisticRegressionModelSchema()

        try:
            # Validate request data
            config = schema.load(request_data)
            
            # Train model and get results
            model_info = ml_classification_service.train_logistic_regression(dataset_id, config)
            
            return jsonify(model_info), 201
        
        except ValidationError as err:
            return jsonify(err.messages), 400
        except ValueError as err:
            return jsonify({"error": str(err)}), 400
        except Exception as err:
            return jsonify({"error": "Internal server error"}), 500

@ml_routes.route('/logistic-regression/<model_id>/plot-data', methods=['GET'])
def get_logistic_regression_model_plot_data(model_id):
    """Get plot data for logistic regression model"""
    try:
        plot_data = ml_classification_service.get_logistic_regression_plot_data(model_id)

        return jsonify(plot_data), 200
    except Exception as err:
        return jsonify({"error": "Internal server error"}), 500

@ml_routes.route('/logistic-regression/predict/<model_id>', methods=['POST'])
def predict_logistic_regression(model_id):
    """Make predictions using a trained logistic regression model"""
    try:
        feature_data = request.json.get("feature_data")
        
        if not feature_data:
            return jsonify({"error": "Missing feature_data"}), 400
            
        predictions = ml_classification_service.predict(model_id, feature_data)
        return jsonify(predictions), 200
        
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    except Exception as err:
        return jsonify({"error": "Internal server error"}), 500
