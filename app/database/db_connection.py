from pymongo import MongoClient
from app.config import Config

# Patrón Singleton para manejar la conexión a MongoDB: garantizando que solo exista
# una instancia de la conexión durante toda la vida útil de la aplicación
class MongoDBConnection:
    _db_client = None
    _db_instance = None

    @classmethod
    def init_db(cls):
        if cls._db_client is None:
            cls._db_client = MongoClient(Config.MONGO_URI)
            cls._db_instance = cls._db_client[Config.MONGO_DB_NAME]
        print("Successfully connected to MongoDB.")

    @classmethod
    def get_db(cls):
        if cls._db_instance is None:
            cls.init_db()
        return cls._db_instance
