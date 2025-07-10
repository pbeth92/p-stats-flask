from marshmallow import Schema, fields, post_dump

from app.shared.schema_utils import SchemaUtils

class BoxPlotDataPointSchema(Schema):
    categorical_label = fields.Str(required=False)
    values = fields.List(fields.Float(), required=True)

class BoxPlotMetadataSchema(Schema):
    total_records = fields.Int(required=True)
    selected_records = fields.Int(required=True)
    numeric_variable = fields.Str(required=True)
    categorical_variable = fields.Str(required=False, allow_none=True)

class BoxPlotSchema(Schema):
    data = fields.List(fields.Nested(BoxPlotDataPointSchema()), required=True)
    data_label = fields.Str(required=True)
    metadata = fields.Nested(BoxPlotMetadataSchema(), required=True)

    @post_dump
    def format_fields(self, data, **kwargs):
        data = { SchemaUtils.to_camel_case(key): SchemaUtils.recursive_camelized_element(value, **kwargs) for key, value in data.items() }
        return data