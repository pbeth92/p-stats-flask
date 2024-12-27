import pickle
from bson import Binary
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from app.services import ModelService, DatasetService

def create_linear_regression_model(data):
    dataset_id = data["dataset_id"]
    test_size = data["test_size"]
    predictors = data["predictors"]
    target = data["target"]

    df = DatasetService.get_dataset_as_dataframe(dataset_id)

    X_train = df[predictors]
    y_train = df[target]
    X_test, y_test = None, None

    if (test_size > 0):
        X_train, X_test, y_train, y_test = train_test_split(
            X_train,
            y_train,
            test_size=test_size,
            random_state=42
        )

    # Entrenar el modelo de regresión lineal
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Serializar el modelo
    model_binary = Binary(pickle.dumps(reg))

    model_document = {
        "target": target,
        "predictors": predictors,
        "model": model_binary,
        "model_type": "linear_regression",
        "dataset_id": dataset_id,
        "X_train": X_train.to_numpy().tolist(),
        "y_train": y_train.tolist(),
        "coef": reg.coef_.tolist(), # Pendiente
        "intercept": reg.intercept_, # Intercepto
        "score": reg.score(X_train, y_train), # Coeficiente de determinación (R²)
    }

    # Save model
    return ModelService.save_model(model_document)

def linear_regression_predict(model_id, values):
    model = ModelService.get_model(model_id)
    if not model:
        raise ValueError(f"Model not found with ID: {model_id}")

    try:
        reg = pickle.loads(model["model"])
    except Exception as e:
        raise ValueError(f"Error getting model: {e}")

    try:
        return [reg.predict([val])[0] for val in values]
    except Exception as e:
        raise ValueError(f"Prediction error: {e}")
