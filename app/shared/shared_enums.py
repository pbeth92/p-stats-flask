from enum import Enum

class StatisticType(Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"

class BivariateStatisticType(Enum):
    CATEGORICAL_CATEGORICAL = "categorical-categorical"
    CATEGORICAL_NUMERICAL = "categorical-numerical"
    NUMERICAL_NUMERICAL = "numerical-numerical"

class ModelType(Enum):
    LINEAR_REGRESSION = "linear-regression"
    LOGISTIC_REGRESSION = "logistic-regression"
    # Add more model types as needed