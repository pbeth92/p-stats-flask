import mongoengine as me
from datetime import datetime

# --- Documentos Embebidos para anidar la información ---

class ModelMetrics(me.EmbeddedDocument):
    mse = me.FloatField()
    rmse = me.FloatField()
    r2 = me.FloatField()
    cv_r2_mean = me.FloatField(required=False) # Puede no estar siempre
    cv_r2_std = me.FloatField(required=False)  # Puede no estar siempre

class Metrics(me.EmbeddedDocument):
    train = me.EmbeddedDocumentField(ModelMetrics)
    test = me.EmbeddedDocumentField(ModelMetrics)

class Coefficients(me.EmbeddedDocument):
    intercept = me.FloatField()
    # Usamos un DictField para guardar los coeficientes de cada variable
    features = me.DictField()

# --- Modelo Principal que se guarda en la colección 'models' ---

class MachineLearningModel(me.Document):
    name = me.StringField(required=True, max_length=100)
    description = me.StringField(max_length=500)
    dataset = me.ObjectIdField(required=True)
    model_type = me.StringField(required=True)
    created_at = me.DateTimeField(default=datetime.utcnow)
    
    feature_variables = me.ListField(me.StringField(), required=True)
    target_variable = me.StringField(required=True)
    
    # Este campo es un diccionario. Asegúrate de que en la lógica de entrenamiento
    # se le asigne un diccionario con los hiperparámetros antes de guardar.
    hyperparameters = me.DictField()
    
    coefficients = me.EmbeddedDocumentField(Coefficients, required=True)
    metrics = me.EmbeddedDocumentField(Metrics, required=True)

    meta = {
        'collection': 'models',
        'allow_inheritance': True # Permite usar este modelo como base para otros
    }
