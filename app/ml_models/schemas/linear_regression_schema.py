from marshmallow import Schema, fields

class CreateLinearRegressionModelSchema(Schema):
    name = fields.Str(required=True)
    description = fields.Str(required=False, allow_none=True)
    target_variable = fields.Str(required=True, data_key="targetVariable")
    feature_variables = fields.List(fields.Str(), required=True, data_key="featureVariables")
    test_size = fields.Float(required=True, data_key="testSize")
    random_state = fields.Int(required=False, data_key="randomState", missing=42, default=42)
    cross_validation_folds = fields.Int(required=False, data_key="crossValidationFolds", missing=5, default=5)
    hyperparameters = fields.Dict(required=False, data_key="hyperparameters", missing={})


