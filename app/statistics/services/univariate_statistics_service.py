from typing import List
from bson import ObjectId
import pandas as pd
import numpy as np
import logging

from app.statistics.models import Statistics
from app.statistics.models.univariate_statistics_model import CategoryFrequencyRecord, IntervalFrequencyRecord, UnivariateStatistics, NumericalStatistics,  CategoricalStatistics
from app.shared.shared_enums import StatisticType

class UnivariateStatisticsService:
    MAX_UNIQUE_CATEGORIES = 100
    MAX_VARIABLES = 50

    def get_univariate_statistics(self, dataset_id: ObjectId):
        try:
            statistics = Statistics.objects(dataset_id=dataset_id).only("univariate_statistics").first()
            
            return statistics.univariate_statistics

        except Exception as e:
            raise ValueError(f"Failed getting univariate statistics from database: {e}")
    
    def generate_univariate_statistics(self, df) -> List[UnivariateStatistics]:
        try:
            # Limit to first MAX_VARIABLES columns if DataFrame has more variables
            if len(df.columns) > self.MAX_VARIABLES:
                df = df.iloc[:, :self.MAX_VARIABLES]
            
            univariate_statistics = []

            for column in df.columns:
                column_data = df[column]
                is_numeric = pd.api.types.is_numeric_dtype(column_data)

                if is_numeric:
                    frequency_table = self._generate_numeric_frequency_table(column_data)
                    descriptive_statistics = self._generate_univariate_descriptive_statistics(column_data, True)

                    univariate_object = UnivariateStatistics(
                        variable=column,
                        type=StatisticType.NUMERICAL,
                        frequency_table=frequency_table,
                        statistical_summary=descriptive_statistics
                    )

                else:
                    frequency_table = self._generate_frecuency_table(column_data)
                    descriptive_statistics = self._generate_univariate_descriptive_statistics(column_data, False)

                    univariate_object = UnivariateStatistics(
                        variable=column,
                        type=StatisticType.CATEGORICAL,
                        frequency_table=frequency_table,
                        statistical_summary=descriptive_statistics
                    )

                univariate_statistics.append(univariate_object)

            return univariate_statistics
        
        except ValueError as ve:
            logging.error(f"Error: {ve}")
            return {"error": str(ve)}
            
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return {"error": "An unexpected error occurred"}
        
    def _generate_univariate_descriptive_statistics(self, column_data: pd.Series, is_numeric: bool) -> dict:
        if is_numeric:
            return NumericalStatistics(
                mean=self._convert_to_native(column_data.mean()),
                median=self._convert_to_native(column_data.median()),
                mode=self._convert_to_native(column_data.mode()[0]),
                std=self._convert_to_native(column_data.std()),
                variance=self._convert_to_native(column_data.var()),
                min=self._convert_to_native(column_data.min()),
                max=self._convert_to_native(column_data.max()),
                percentile_25=self._convert_to_native(column_data.quantile(0.25)),
                percentile_75=self._convert_to_native(column_data.quantile(0.75)),
                skewness=self._convert_to_native(column_data.skew()),
                kurtosis=self._convert_to_native(column_data.kurt())
            )
        
        else:
            unique_categories = column_data.nunique()
            value_counts = column_data.value_counts()

            if unique_categories > self.MAX_UNIQUE_CATEGORIES:
                return {}

            return CategoricalStatistics(
                unique_categories=unique_categories,
                most_frequent_category=value_counts.idxmax(),
                frequency={str(k): v for k, v in value_counts.to_dict().items()},
            )
    
    @staticmethod #para métodos auxiliares sin dependencia de la clase o instancia
    def _convert_to_native(value):
        if isinstance(value, (np.integer, np.int64)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64)):
            return float(value)
        return value

    @staticmethod
    def _generate_frecuency_table(column_data: pd.Series) -> List[IntervalFrequencyRecord]:
        # Frecuencia absoluta: Número de ocurrencias de cada categoría
        frequency_table = column_data.value_counts().reset_index()
        frequency_table.columns = ['category', 'absolute_frequency']

        # Frecuencia relativa: Porcentaje de la frecuencia de cada categoría en relación al total.
        total_count = frequency_table['absolute_frequency'].sum()
        frequency_table['relative_frequency'] = frequency_table['absolute_frequency'] / total_count

        # Frecuencia absoluta acumulada
        frequency_table['cumulative_absolute_frequency'] = frequency_table['absolute_frequency'].cumsum()

        # Frecuencia relativa acumulada
        frequency_table['cumulative_relative_frequency'] = frequency_table['relative_frequency'].cumsum() * 100

        category_records = [
            CategoryFrequencyRecord(
                category=row['category'],
                absolute_frequency=row['absolute_frequency'],
                relative_frequency= row['relative_frequency'],
                cumulative_absolute_frequency=row['cumulative_absolute_frequency'],
                cumulative_relative_frequency=row['cumulative_relative_frequency']
            )
            for _, row in frequency_table.iterrows()
        ]

        return category_records

    @staticmethod
    def _generate_numeric_frequency_table(column_data: pd.Series, method: str ='sturges') -> List[IntervalFrequencyRecord]:
        # Número de clases
        num_classes = int(1 + 3.3 * np.log10(len(column_data)))

        # Definición de los intervalos de clase usando el rango de los datos
        min_value = column_data.min()
        max_value = column_data.max()
        bins = np.linspace(min_value, max_value, num_classes + 1)  # Genera los límites de los intervalos
        labels = [f"{round(bins[i], 2)} - {round(bins[i+1], 2)}" for i in range(num_classes)]

        # Freuncia absoluta ni
        frequency_table = pd.cut(column_data, bins=bins, labels=labels, include_lowest=True).value_counts().sort_index().reset_index()
        frequency_table.columns = ['interval', 'absolute_frequency']

        # Marca de clase xi
        frequency_table['class_mark'] = [(bins[i] + bins[i+1]) / 2 for i in range(num_classes)]

        # Frecuencia relativa
        total_count = frequency_table['absolute_frequency'].sum()
        frequency_table['relative_frequency'] = (frequency_table['absolute_frequency'] / total_count)
        
        # Frecuencia absoluta acumulada Ni
        frequency_table['cumulative_absolute_frequency'] = frequency_table['absolute_frequency'].cumsum()
        
        # Frecuencia relativa acumulada
        frequency_table['cumulative_relative_frequency'] = frequency_table['relative_frequency'].cumsum()

        interval_records = [
            IntervalFrequencyRecord(
                interval=row['interval'],
                class_mark=row['class_mark'],
                absolute_frequency=int(row['absolute_frequency']),
                relative_frequency=float(row['relative_frequency']),
                cumulative_absolute_frequency=int(row['cumulative_absolute_frequency']),
                cumulative_relative_frequency=float(row['cumulative_relative_frequency'])
            )
            for _, row in frequency_table.iterrows()
        ]

        return interval_records