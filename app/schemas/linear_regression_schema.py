from marshmallow import fields, pre_load
import humps

from app.schemas.ml_base import MLBaseSchema

class LinearRegressionSchema(MLBaseSchema):
    variable_x = fields.String(required=True)
    variable_y = fields.List(fields.String, required=True)

    @pre_load
    def to_snake_case(self, data, **kwargs):
        return humps.decamelize(data)