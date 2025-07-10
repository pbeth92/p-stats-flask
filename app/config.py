import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGO_DB_USER = os.getenv('MONGO_DB_USER')
    MONGO_DB_PASSWORD = os.getenv('MONGO_DB_PASSWORD')
    MONGO_DB_NAME = os.getenv('MONGO_DB_NAME')
    MONGO_DB_PORT = os.getenv('MONGO_DB_PORT')
    MONGO_DB_HOST = os.getenv('MONGO_DB_HOST')
    MONGO_URI = f"mongodb://{MONGO_DB_USER}:{MONGO_DB_PASSWORD}@{MONGO_DB_HOST}:{MONGO_DB_PORT}/"

    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', False)