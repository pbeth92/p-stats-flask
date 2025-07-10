from mongoengine import Document, StringField, ReferenceField, DictField, EnumField, ListField, DateTimeField
from datetime import datetime
from app.shared.shared_enums import ModelType
from app.dataset.models.dataset_model import Dataset

class MLModel(Document):
    name = StringField(required=True)
    description = StringField()
    dataset = ReferenceField(Dataset, required=True)
    model_type = EnumField(ModelType, required=True)
    created_at = DateTimeField(default=datetime.utcnow)
    
    # Model configuration
    feature_variables = ListField(StringField(), required=True)
    target_variable = StringField(required=True)
    hyperparameters = DictField()
    
    # Model parameters
    coefficients = DictField(required=True)  # For linear models: feature -> coefficient
    intercept = StringField(required=True)   # Stored as string to maintain precision
    
    # Model performance
    metrics = DictField(required=True)       # Contains train/test metrics
    
    meta = {
        'collection': 'ml_models',
        'indexes': [
            'dataset',
            'model_type',
            'created_at'
        ]
    }
    
    def to_dict(self):
        """Convert model to dictionary for API responses"""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'dataset_id': str(self.dataset.id),
            'model_type': self.model_type.value,
            'created_at': self.created_at.isoformat(),
            'feature_variables': self.feature_variables,
            'target_variable': self.target_variable,
            'hyperparameters': self.hyperparameters,
            'coefficients': self.coefficients,
            'intercept': self.intercept,
            'metrics': self.metrics
        }