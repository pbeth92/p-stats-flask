from pymongo.errors import PyMongoError
from marshmallow import ValidationError
from bson import ObjectId
import logging

from app.database.db_connection import MongoDBConnection
from app.schemas.model_document_schema import LinearRegressionDocumentSchema

class ModelService:
    @classmethod
    def _get_coll(cls):
        db = MongoDBConnection.get_db()
        return db['models']

    @classmethod
    def save_model(cls, model_document: dict) -> ObjectId:
        try:
            schema = LinearRegressionDocumentSchema()
            validated_data = schema.dump(model_document)

            # Convertir dataset_id a ObjectId si es necesario
            if isinstance(validated_data["dataset_id"], str):
                validated_data["dataset_id"] = ObjectId(validated_data["dataset_id"])

            result = cls._get_coll().insert_one(validated_data)
            logging.info(f"Model saved successfully with ID {result.inserted_id}.")
            return result.inserted_id
        except ValidationError as ve:
            logging.error(f"Validation error: {ve.messages}")
            raise ValueError(f"Validation error: {ve.messages}")
        except PyMongoError as e:
            logging.error(f"An error occurred while saving the model: {e}")
            raise ValueError(f"Failed to save model to database: {e}")

    @classmethod
    def find_models(cls):
        try:
            return cls._get_coll().find()
        except PyMongoError as e:
            logging.error(f"An error occurred while retrieving models: {e}")
            raise ValueError("Failed to retrieve models.")

    @classmethod
    def find_model_by_type(cls, dataset_id, type, fields):
        try:
            projection = {field: 1 for field in fields} if fields else None

            return cls._get_coll().find(
                {
                    "dataset_id": ObjectId(dataset_id),
                    "model_type": type
                },
                projection=projection
            )
        except PyMongoError as e:
            logging.error(f"An error occurred while retrieving models: {e}")
            raise ValueError("Failed to retrieve models.")

    @classmethod
    def find_model_by_id(cls, model_id: str) -> dict:
        try:
            return cls._get_coll().find_one({"_id": ObjectId(model_id)})
        except PyMongoError as e:
            logging.error(f"An error occurred while retrieving the model: {e}")
            raise ValueError("Failed to retrieve the model.")
        except Exception as ex:
            logging.error(f"Invalid ObjectId: {model_id} - {ex}")
            raise ValueError(f"Invalid model ID: {model_id}")
