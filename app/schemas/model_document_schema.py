import mongoengine as me
from marshmallow_mongoengine import ModelSchema

supported_models = ["linear_regression"]

class MLModelDocument(me.Document):
    model = me.BinaryField(required=True)
    model_type = me.StringField(required=True, choices=supported_models)
    dataset_id = me.ObjectIdField(required=True)

    meta = {
        'abstract': True
    }

class LinearRegressionDocument(MLModelDocument):
    target = me.StringField(required=True)
    predictors = me.ListField(me.StringField(), required=True)
    X_train = me.ListField(me.ListField(me.FloatField()), required=True)
    y_train = me.ListField(me.FloatField(), required=True)
    intercept = me.FloatField(required=True)
    coef = me.ListField(me.FloatField(), required=True)
    score = me.FloatField(required=True)

    meta = {
        'collection': 'models'
    }

class ModelDocumentSchema(ModelSchema):
    class Meta:
        model = MLModelDocument

class LinearRegressionDocumentSchema(ModelSchema):
    class Meta:
        model = LinearRegressionDocument