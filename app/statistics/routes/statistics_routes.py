from bson import ObjectId
from flask import Blueprint, jsonify, request

from app.statistics.services.statistics_service import StatisticsService
from app.statistics.schemas.univariate_statistics_schema import UnivariateStatisticsSchema
from app.statistics.schemas.bivariate_statistics_schema import BivariateStatisticsSchema
from app.statistics.schemas.statistics_schema import BoxPlotSchema
from app.dataset.services.dataset_service import UnivariateStatisticsService
from app.dataset.services.dataset_service import BivariateStatisticsService

statistics_routes = Blueprint('statistics', __name__, url_prefix='/statistics')
statistics_service = StatisticsService()
univariate_statisticss_service = UnivariateStatisticsService()
bivariate_statisticss_service = BivariateStatisticsService()

@statistics_routes.route('/univariate/<dataset_id>', methods=['GET'])
def get_univariate_stats(dataset_id):
    try:
        if not ObjectId.is_valid(dataset_id):
            return jsonify({
                "error": "Validation error",
                "details": "Invalid dataset id"
            }), 400
        
        univariate_statistics = univariate_statisticss_service.get_univariate_statistics(
            ObjectId(dataset_id)
        )

        if univariate_statistics is None:
            return jsonify({
                "error": f"No univariate statistics found for dataset {dataset_id}"
            }), 404
        
        schema = UnivariateStatisticsSchema(many=True)
        result = schema.dump(univariate_statistics)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": f"Internal server error: {e}"}), 500
    
@statistics_routes.route('/bivariate/<dataset_id>', methods=['GET'])
def get_bivariate_stats(dataset_id):
    try:
        if not ObjectId.is_valid(dataset_id):
            return jsonify({
                "error": "Validation error",
                "details": "Invalid dataset id"
            }), 400
        
        bivariate_statistics = bivariate_statisticss_service.get_bivariate_statistics(
            ObjectId(dataset_id)
        )

        if bivariate_statistics is None:
            return jsonify({
                "error": f"No univariate statistics found for dataset {dataset_id}"
            }), 404
        
        schema = BivariateStatisticsSchema(many=True)
        result = schema.dump(bivariate_statistics)
        
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": f"Internal server error: {e}"}), 500
    
@statistics_routes.route('/box-plot/<dataset_id>/<numerical_variable>', methods=['GET'])
def dataset_box_plot(dataset_id, numerical_variable):
    try:
        if not ObjectId.is_valid(dataset_id):
            return jsonify({
                "error": "Validation error",
                "details": "Invalid dataset id"
            }), 400

        category_var = request.args.get('category_var', default=None)
        subset_size = request.args.get('subset_size', type=int, default=None)

        result = statistics_service.get_box_plot_data(
            dataset_id,
            numerical_variable,
            category_var,
            subset_size
        )

        # Serialize response using BoxPlotSchema
        schema = BoxPlotSchema()
        serialized_result = schema.dump(result)
        
        return jsonify(serialized_result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {e}"}), 500