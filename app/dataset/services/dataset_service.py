import pandas as pd
import logging
import threading
from bson import ObjectId
from mongoengine import QuerySet, ValidationError
from datetime import datetime, timezone

from app.dataset.models.dataset_model import Dataset, Variable
from app.dataset.services.dataset_records_service import DatasetRecordsService
from app.dataset.utils.utils import DatasetStatus
from app.statistics.models.bivariate_statistics_model import BivariateStatistics
from app.statistics.services.bivariate_statistics_service import BivariateStatisticsService
from app.statistics.services.statistics_service import StatisticsService
from app.statistics.services.univariate_statistics_service import UnivariateStatisticsService
from app.statistics.models import Statistics

class DatasetService:
    def __init__(self):
        self.dataset_records_service = DatasetRecordsService()
        self.univariate_statistics_service = UnivariateStatisticsService()
        self.bivariate_statistics_service = BivariateStatisticsService()
        self.statistics_service = StatisticsService()
    
    def create_dataset(self, dataset_schema) -> Dataset:
        file = dataset_schema['file']

        if not file or file.filename == '':
            raise ValueError("No file uploaded or invalid file.")

        try:
            if file.filename.endswith('.csv'):
                delimiters = [',', ';', '\t', '|']
                df = None

                for delimiter in delimiters:
                    try:
                        file.seek(0)
                        df = pd.read_csv(file, delimiter=delimiter)
                        
                        if df.shape[1] > 1:
                            break
                    except:
                        continue

                if df is None or df.shape[1] == 1:
                    file.seek(0)
                    df = pd.read_csv(file, sep=None, engine='python')
                
                if df is None:
                    raise ValueError("Could not parse CSV file with any common delimiter")
                
            elif file.filename.endswith('.json'):
                df = pd.read_json(file)
            else:
                raise ValueError("Unsupported file format. Only CSV and JSON are supported.")
            
        except Exception as e:
            raise ValueError(f"Error creating dataframe: {e}")
        
        size_in_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        variables = [
            Variable(name=col, type=str(df[col].dtype))
            for col in df.columns
        ]

        dataset_document = Dataset(
            name=dataset_schema['name'],
            description=dataset_schema['description'],
            tags=dataset_schema.get('tags', []),
            n_cols=df.shape[1],
            n_rows=df.shape[0],
            size=float(size_in_mb),
            variables=variables,
            create_at=datetime.now(timezone.utc),
            update_at=datetime.now(timezone.utc),
            status=DatasetStatus.PROCESSING.value
        )

        dataset = self.save_dataset(dataset_document)

        threading.Thread(target=self.process_dataset, args=(dataset.id, df)).start()

        return dataset
    
    def save_dataset(self, dataset: Dataset) -> Dataset:
        try:
            return dataset.save()
        
        except ValidationError as ve:
            raise ValueError(f"Validation error: {ve.messages}")
        except Exception as e:
            raise ValueError(f"Failed saving dataset to database: {e}")
    
    def process_dataset(self, dataset_id, df):
        try:
            # 1. Save Dataset Records
            self.dataset_records_service.insert_dataset_records(
                dataset_id,
                df.to_dict('records')
            )

            # 2.1 Generate Univariate Statistics
            univariate_statistics = self.univariate_statistics_service.generate_univariate_statistics(df)

            # 2.2 Generate Bivariate Statistics
            bivariate_statistics = self.bivariate_statistics_service.generate_bivariate_statistics(df)

            # Save Statistics
            statistics = Statistics(
                dataset_id=dataset_id,
                univariate_statistics=univariate_statistics,
                bivariate_statistics=bivariate_statistics
            )

            self.statistics_service.save_statistics(statistics)

            print(f"Completed processing dataset {dataset_id}")
            self.update_dataset_status(dataset_id, DatasetStatus.COMPLETED)
        
        except Exception as e:
            print(f"Error processing dataset {dataset_id}: {str(e)}")
            self.update_dataset_status(dataset_id, DatasetStatus.FAILED)
        
    def get_dataset_by_id(self, id: ObjectId):
        try:
            return Dataset.objects(id=id).first()
        except ValidationError as e:
            raise ValueError(f"Failed getting dataset from database: {e}")
        
    
    def get_datasets(self) -> QuerySet:
        try:
            return Dataset.objects()
        except ValidationError as e:
            raise ValueError(f"Failed getting datasets from database: {e}")
        
    def update_dataset_status(self, id: ObjectId, status: DatasetStatus) -> QuerySet:
        try:
            Dataset.objects(id=id).update(status=status.value, update_at=datetime.now(timezone.utc))
        except ValidationError as e:
            raise ValueError(f"Failed updating dataset status: {e}")