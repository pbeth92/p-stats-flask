import mongoengine as me
from app.config import Config

# Patrón Singleton para manejar la conexión a MongoDB: garantizando que solo exista
# una instancia de la conexión durante toda la vida útil de la aplicación
class MongoDBConnection:
    @classmethod
    def init_db(cls):
        try:
            me.connect(
                db=Config.MONGO_DB_NAME,
                host=Config.MONGO_URI
            )
            
            print("Successfully connected to MongoDB.")

        except Exception as e:
            print(f"Error connecting to MongoDB: {str(e)}")

    @classmethod
    def get_db(cls):
        if not me.connection:
            cls.init_db()
        return me.connection
