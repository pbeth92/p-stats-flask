from flask import Flask
from app.routes.ml_routes import ml_routes
from app.database.db_connection import MongoDBConnection

def create_app():
    app = Flask(__name__)

    app.register_blueprint(ml_routes)

    return app

def init_db():
    MongoDBConnection.init_db()