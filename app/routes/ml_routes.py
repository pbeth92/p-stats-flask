from flask import Blueprint

ml_routes = Blueprint('ml', __name__, url_prefix='/ml')

@ml_routes.route('/linear-regression', methods=['POST'])
def linear_regression():
    return 'test'