from mongoengine import ValidationError
import random
from app.dataset.models.dataset_model import create_dataset_record_model

class DatasetRecordsService:
    def insert_dataset_records(self, dataset_id, records):
        try:
            DatasetRecord = create_dataset_record_model(str(dataset_id))

            dataset_records = []
            for record in records:
                dataset_record = DatasetRecord(**record)
                dataset_records.append(dataset_record)

            DatasetRecord.objects.insert(dataset_records)
        
        except ValidationError as ve:
            raise ValueError(f"Validation error: {ve.messages}")
        except Exception as e:
            raise ValueError(f"Failed saving dataset records to database: {e}")
    
    def get_all_dataset_records(self, dataset_id, limit=None, skip=None):
        try:
            DatasetRecord = create_dataset_record_model(dataset_id)
            
            # Get total count
            total = DatasetRecord.objects().count()
            
            # Get base query
            query = DatasetRecord.objects()
            
            # Apply pagination only if both parameters are provided
            if limit is not None and skip is not None:
                records = query.skip(skip).limit(limit)
                metadata = {
                    'total': total,
                    'limit': limit,
                    'skip': skip
                }
            else:
                records = query
                metadata = {
                    'total': total
                }
            
            return {
                'records': records,
                'metadata': metadata
            }

        except Exception as e:
            raise ValueError(f"Failed getting dataset records from database: {e}")
        
    def get_records_by_variables(self, dataset_id, variables, subset_size=None):
        """
        Get dataset records filtering by specified variables
        If subset_size is provided, returns a random sample of that size
        """
        try:
            DatasetRecord = create_dataset_record_model(dataset_id)
            
            # Get base query with projection
            query = DatasetRecord.objects().only(*variables)
            
            if subset_size is not None:
                # Use limit and skip for random sampling instead of loading all records
                total = query.count()
                if total > subset_size:
                    skip = random.randint(0, total - subset_size)
                    records = query.skip(skip).limit(subset_size)
                else:
                    records = query
            else:
                records = query
            
            return {
                'records': records
            }
            
        except Exception as e:
            raise ValueError(f"Failed getting dataset records by variables: {e}")
