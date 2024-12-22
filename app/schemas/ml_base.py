from marshmallow import Schema, fields

class MLBaseSchema(Schema):
    test_size = fields.Float(missing=0.2, validate=lambda x: x > 0 and x < 1)
    cv = fields.Integer(missing=5, validate=lambda x: x > 0)