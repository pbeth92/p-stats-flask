from bson import ObjectId
from flask import Blueprint, request, jsonify
from marshmallow import ValidationError

from app.dataset.schemas.dataset_schema import CreateDatasetSchema, DatasetSchema, create_dynamic_schema
from app.dataset.services.dataset_service import DatasetService
from app.dataset.services.dataset_records_service import DatasetRecordsService

dataset_routes = Blueprint('dataset', __name__, url_prefix='/dataset')
dataset_service = DatasetService()
dataset_records_service = DatasetRecordsService()

@dataset_routes.route('', methods=['POST', 'GET'])
def dataset():
    if request.method == 'POST':
        request_data = {
            'name': request.form.get('name'),
            'description': request.form.get('description'),
            'tags': request.form.getlist('tags'),
            'file': request.files.get('file')
        }

        try:
            schema = CreateDatasetSchema()
            data = schema.load(request_data)
        
        except ValidationError as err:
            return jsonify({
                "error": "Validation error",
                "details": err.messages
            }), 400
    
        try:
            dataset = dataset_service.create_dataset(data)
            schema = DatasetSchema()
            result = schema.dump(dataset)
            return jsonify(result), 201

        except ValueError as e:
            return jsonify({"error": str(e)}), 500
        except Exception as e:
            return jsonify({"error": "Unexpected error occurred"}), 500

    if request.method == 'GET':
        datasets = dataset_service.get_datasets()
        schema = DatasetSchema(many=True)
        result = schema.dump(datasets)
        return jsonify(result), 200
    
@dataset_routes.route('/<dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    try:
        if not ObjectId.is_valid(dataset_id):
            return jsonify({
                "error": "Validation error",
                "details": "Invalid dataset id"
            }), 400
        
        dataset = dataset_service.get_dataset_by_id(
            ObjectId(dataset_id)
        )

        if dataset is None:
            return jsonify({
                "error": f"No dataset found with id {dataset_id}"
            }), 404
        
        schema = DatasetSchema()
        result = schema.dump(dataset)
        
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": f"Internal server error: {e}"}), 500

@dataset_routes.route('/records/<dataset_id>', methods=['GET'])
def dataset_records(dataset_id):
    try:
        if not ObjectId.is_valid(dataset_id):
            return jsonify({
                "error": "Validation error",
                "details": "Invalid dataset id"
            }), 400

        limit = request.args.get('limit', type=int, default=None)
        skip = request.args.get('skip', type=int, default=None)

        if limit is not None and limit < 1:
            return jsonify({
                "error": "Validation error",
                "details": "Limit must be greater than 0"
            }), 400
        
        if skip is not None and skip < 0:
            return jsonify({
                "error": "Validation error",
                "details": "Skip must be greater than or equal to 0"
            }), 400

        result = dataset_records_service.get_all_dataset_records(dataset_id, limit=limit, skip=skip)
        
        if result['records'].count() == 0:
            return jsonify({
                "error": f"No dataset records found for dataset {dataset_id}"
            }), 404

        schema = create_dynamic_schema(result['records'][0])
        data = schema(many=True).dump(result['records'])

        return jsonify({
            "records": data,
            "metadata": result['metadata']
        }), 200

    except Exception as e:
        return jsonify({"error": f"Internal server error: {e}"}), 500

@dataset_routes.route('/records/<dataset_id>/variables', methods=['GET'])
def dataset_records_by_variables(dataset_id):
    try:
        if not ObjectId.is_valid(dataset_id):
            return jsonify({
                "error": "Validation error",
                "details": "Invalid dataset id"
            }), 400

        # Get variables from query parameters (required)
        variables = request.args.getlist('variables')
        if not variables:
            return jsonify({
                "error": "Validation error",
                "details": "At least one variable must be specified"
            }), 400

        # Get optional subset_size parameter
        subset_size = request.args.get('subset_size', type=int, default=None)
        if subset_size is not None and subset_size < 1:
            return jsonify({
                "error": "Validation error",
                "details": "Subset size must be greater than 0"
            }), 400

        result = dataset_records_service.get_records_by_variables(
            dataset_id,
            variables,
            subset_size=subset_size
        )

        if not result['records']:
            return jsonify({
                "error": f"No dataset records found for dataset {dataset_id}"
            }), 404

        schema = create_dynamic_schema(result['records'][0])
        data = schema(many=True).dump(result['records'])

        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": f"Internal server error: {e}"}), 500
