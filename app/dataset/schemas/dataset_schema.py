import os
from marshmallow import Schema, validate, fields, ValidationError, pre_dump, post_dump

from app.dataset.utils.utils import DatasetStatus
from app.shared.schema_utils import SchemaUtils

allowed_extensions = [".csv", ".json"]

def has_valid_extension(value):
    file_extension = os.path.splitext(value.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise ValidationError("File extension not supported")

class CreateDatasetSchema(Schema):
    name = fields.String(required=True)
    description = fields.String(required=True)
    tags = fields.List(fields.String(), default=[])
    file = fields.Field(required=True, validate=has_valid_extension)

class VariableSchema(Schema):
    name = fields.String(required=True)
    type = fields.String(required=True)
    
class DatasetSchema(Schema):
    id = fields.String(data_key="id", required=True)
    name = fields.String(required=True)
    description = fields.String(required=True)
    tags = fields.List(fields.String(), default=[])
    n_cols = fields.Integer(required=True)
    n_rows = fields.Integer(required=True)
    size = fields.Float(required=True)
    variables = fields.List(fields.Nested(VariableSchema), required=True)
    create_at = fields.DateTime(required=True)
    update_at = fields.DateTime(required=True)
    status = fields.String(required=True, validate=validate.OneOf([e.value for e in DatasetStatus]))

    @pre_dump
    def convert_object_id(self, data, **kwargs):
        if "_id" in data:
            data["id"] = str(data.pop("_id"))
        return data

    @post_dump
    def format_fields(self, data, **kwargs):
        data = {SchemaUtils.to_camel_case(key): value for key, value in data.items()}
        return data
    
def create_dynamic_schema(record):
    record_dict = record.to_mongo()

    class DynamicDatasetRecordSchema(Schema):
        for key in record_dict.keys():
            vars()[key] = fields.Raw()
        
    return DynamicDatasetRecordSchema
