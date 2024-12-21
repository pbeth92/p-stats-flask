from flask import Flask
from app.routes.ml_routes import ml_routes

def create_app():
    app = Flask(__name__)

    app.register_blueprint(ml_routes)

    return app