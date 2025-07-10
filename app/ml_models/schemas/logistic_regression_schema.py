from marshmallow import Schema, fields, post_dump
from app.shared.schema_utils import SchemaUtils

class CreateLogisticRegressionModelSchema(Schema):
    name = fields.Str(required=True)
    description = fields.Str(required=False, allow_none=True)
    target_variable = fields.Str(required=True, data_key="targetVariable")
    feature_variables = fields.List(fields.Str(), required=True, data_key="featureVariables")
    test_size = fields.Float(required=True, data_key="testSize")
    random_state = fields.Int(required=False, data_key="randomState", missing=42, default=42)
    cross_validation_folds = fields.Int(required=False, data_key="crossValidationFolds", missing=5, default=5)
    hyperparameters = fields.Dict(required=False, data_key="hyperparameters", missing={})

class LogisticRegressionModelResponseSchema(Schema):
    id = fields.Str(required=True)
    name = fields.Str(required=True)
    description = fields.Str(allow_none=True)
    dataset_id = fields.Str(required=True, data_key="datasetId")
    model_type = fields.Str(required=True, data_key="modelType")
    created_at = fields.DateTime(required=True, data_key="createdAt")
    feature_variables = fields.List(fields.Str(), required=True, data_key="featureVariables")
    target_variable = fields.Str(required=True, data_key="targetVariable")
    hyperparameters = fields.Dict(required=True)
    coefficients = fields.Dict(required=True)
    intercept = fields.Str(required=True)
    metrics = fields.Dict(required=True)

    @post_dump
    def camelcase_keys(self, data, **kwargs):
        return SchemaUtils.recursive_camelized_element(data)

class LogisticRegressionModelListResponseSchema(Schema):
    models = fields.List(fields.Nested(LogisticRegressionModelResponseSchema), required=True)
    total = fields.Int(required=True)

    @post_dump
    def camelcase_keys(self, data, **kwargs):
        return SchemaUtils.recursive_camelized_element(data)