from mongoengine import Document, StringField, ReferenceField, DictField, EnumField, ListField, DateTimeField, FloatField
from datetime import datetime
from app.shared.shared_enums import ModelType
from app.dataset.models.dataset_model import Dataset

class LogisticRegressionModel(Document):
    """
    Specific model for Logistic Regression with classification-specific fields
    """
    name = StringField(required=True)
    description = StringField()
    dataset = ReferenceField(Dataset, required=True)
    model_type = EnumField(ModelType, required=True, default=ModelType.LOGISTIC_REGRESSION)
    created_at = DateTimeField(default=datetime.utcnow)
    
    # Model configuration
    feature_variables = ListField(StringField(), required=True)
    target_variable = StringField(required=True)
    hyperparameters = DictField()
    
    # Model parameters (specific to logistic regression)
    coefficients = DictField(required=True)  # For logistic models: feature -> coefficient
    intercept = FloatField(required=True)    # Intercept value
    
    # Classification performance metrics
    train_metrics = DictField(required=True)  # Training metrics
    test_metrics = DictField(required=True)   # Test metrics
    
    # Cross-validation results
    cv_scores = DictField()  # Cross-validation scores
    
    meta = {
        'collection': 'logistic_regression_models',
        'indexes': [
            'dataset',
            'model_type',
            'created_at',
            'target_variable'
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
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
            'cv_scores': self.cv_scores
        }
    
    def get_feature_importance(self):
        """
        Get feature importance based on absolute coefficient values
        """
        feature_importance = {}
        for feature, coef in self.coefficients.items():
            feature_importance[feature] = abs(float(coef))
        
        # Sort by importance (descending)
        sorted_importance = dict(sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True))
        return sorted_importance
    
    def predict_probability(self, feature_values):
        """
        Calculate prediction probability using the sigmoid function
        """
        import numpy as np
        
        # Calculate linear combination
        z = self.intercept
        for feature, value in feature_values.items():
            if feature in self.coefficients:
                z += float(self.coefficients[feature]) * value
        
        # Apply sigmoid function
        probability = 1 / (1 + np.exp(-z))
        return probability
    
    def predict_class(self, feature_values, threshold=0.5):
        """
        Predict class based on probability threshold
        """
        probability = self.predict_probability(feature_values)
        return 1 if probability >= threshold else 0