from marshmallow import Schema, fields, pre_load, pre_dump
import humps

from app.schemas.ml_base import MLBaseSchema

class LinearRegressionSchema(MLBaseSchema):
    target = fields.String(required=True)
    predictors = fields.List(fields.String(), required=True)

    @pre_load
    def to_snake_case(self, data, **kwargs):
        return humps.decamelize(data)
    
class LinearRegressionPredictSchema(Schema):
    values = fields.List(fields.List(fields.Float), required=True)

class LinearRegressionResponseSchema(Schema):
    id = fields.Str(data_key="id", required=True)
    model_type = fields.Str(required=True)
    target = fields.Str(required=True)
    predictors = fields.List(fields.Str(), required=True)

    @pre_dump
    def convert_object_id(self, data, **kwargs):
        if "_id" in data:
            data["id"] = str(data.pop("_id"))
        return data