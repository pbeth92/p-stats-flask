from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression

def create_linear_regression_model(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create a Linear Regression model
    model = LinearRegression()

    # Define the hyperparameters to tune