import pandas as pd

from app.database.db_connection import MongoDBConnection

class DatasetService:
    @classmethod
    def _get_coll(cls, id):
        db = MongoDBConnection.get_db()
        coll = f'{id}_dataset_rows'
        return db[coll]

    @classmethod
    def get_dataset_as_dataframe(cls, id):
        try:        
            dataset = list(cls._get_coll(id).find({}, {"_id": 0}))
            if dataset:
                df = pd.DataFrame(dataset)
                return df
            else:
                print(f"No dataset found with id: {id}")
                return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None