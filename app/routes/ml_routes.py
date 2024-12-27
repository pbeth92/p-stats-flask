from flask import Blueprint, request, jsonify
from marshmallow import ValidationError
import logging

from app.schemas.linear_regression_schema import LinearRegressionSchema, LinearRegressionPredictSchema, LinearRegressionResponseSchema
from app.utils.ml_regression import create_linear_regression_model as create, linear_regression_predict as predict
from app.services import ModelService

ml_routes = Blueprint('ml', __name__, url_prefix='/ml')

@ml_routes.route('/<model_id>', methods=['GET'])
def get_linear_regression_model(model_id):
    try:
        model = ModelService.find_model_by_id(model_id)
        if not model:
            return jsonify({"error": "Model not found"}), 404

        schema = LinearRegressionResponseSchema()
        result = schema.dump(model)

        return jsonify(result), 200
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 500

@ml_routes.route('/linear-regression', methods=['POST', 'PUT'])
def linear_regression():
    if request.method == 'POST':
        request_data = request.json
        schema = LinearRegressionSchema()

        try:
            data = schema.load(request_data)
            model_id = create(data)
        
        except ValidationError as err:
            return jsonify(err.messages), 400
    
        return {"id": str(model_id)}, 201
    else:
        return ''
        
@ml_routes.route('/linear-regression/predict/<model_id>', methods=['POST'])
def linear_regression_predict(model_id):
    request_data = request.json
    schema = LinearRegressionPredictSchema()

    try:
        data = schema.load(request_data)
        predictions = predict(model_id, data["values"])
        return jsonify(predictions), 200

    except ValidationError as err:
        return jsonify({"error": "Validation error", "details": err.messages}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500
