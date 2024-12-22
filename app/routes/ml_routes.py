from flask import Blueprint, request, jsonify
from marshmallow import ValidationError

from app.schemas.linear_regression_schema import LinearRegressionSchema

ml_routes = Blueprint('ml', __name__, url_prefix='/ml')

@ml_routes.route('/linear-regression', methods=['POST'])
def linear_regression():
    request_data = request.json
    schema = LinearRegressionSchema()

    try:
        data = schema.load(request_data)
    except ValidationError as err:
        return jsonify(err.messages), 400
    
    return data