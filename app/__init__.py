from flask import Flask
from flask_cors import CORS
from app.dataset.routes.dataset_routes import dataset_routes
from app.statistics.routes.statistics_routes import statistics_routes
from app.ml_models.routes.ml_routes import ml_routes
from app.database.db_connection import MongoDBConnection

def create_app():
    app = Flask(__name__)
    CORS(app)

    app.register_blueprint(statistics_routes)
    app.register_blueprint(ml_routes)
    app.register_blueprint(dataset_routes)

    return app

def init_db():
    MongoDBConnection.init_db()