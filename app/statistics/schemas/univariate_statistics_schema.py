from marshmallow import Schema, fields, post_dump, pre_dump

from app.shared.schema_utils import SchemaUtils
from app.shared.shared_enums import StatisticType
    
class IntervalFrequencySchema(Schema):
    interval = fields.Str(required=True)
    class_mark = fields.Float(required=True)
    absolute_frequency = fields.Int(required=True)
    relative_frequency = fields.Float(required=True)
    cumulative_absolute_frequency = fields.Int(required=True)
    cumulative_relative_frequency = fields.Float(required=True)

class CategoryFrequencySchema(Schema):
    category = fields.Str(required=True)
    absolute_frequency = fields.Int(required=True)
    relative_frequency = fields.Float(required=True)
    cumulative_absolute_frequency = fields.Int(required=True)
    cumulative_relative_frequency = fields.Float(required=True)

class CategoricalStatisticsSchema(Schema):
    unique_categories = fields.Int(required=True)
    most_frequent_category = fields.Str(required=True)
    frequency = fields.Dict(keys=fields.Str(), values=fields.Int(), required=True)

class NumericalStatisticsSchema(Schema):
    mean = fields.Float(required=True)
    median = fields.Float(required=True)
    mode = fields.Float(required=True)
    std = fields.Float(required=True)
    variance = fields.Float(required=True)
    min = fields.Float(required=True)
    max = fields.Float(required=True)
    percentile_25 = fields.Float(required=True)
    percentile_75 = fields.Float(required=True)
    skewness = fields.Float(required=True)
    kurtosis = fields.Float(required=True)

class UnivariateStatisticsSchema(Schema):
    variable = fields.Str(required=True)
    type = fields.Enum(
        enum=StatisticType,
        by_value=True,
        required=True
    )
    frequency_table = fields.List(fields.Raw(), required=True)
    statistical_summary = fields.Raw(required=True)

    @pre_dump
    def validate_and_flatten(self, data, **kwargs):
        """Este método asegura que frequency_table tiene el esquema correcto
        dependiendo del tipo (NUMERIC o CATEGORICAL)"""

        if data["type"].value == StatisticType.NUMERICAL.value:
            # Serializar la frecuencia en caso de ser tipo NUMERIC
            data["frequency_table"] = IntervalFrequencySchema(many=True).dump(data["frequency_table"])

            # Serializar la estadística en caso de ser tipo NUMERIC
            data["statistical_summary"] = NumericalStatisticsSchema().dump(data["statistical_summary"])

        elif data["type"].value == StatisticType.CATEGORICAL.value:
            # Serializar la frecuencia en caso de ser tipo CATEGORICAL
            data["frequency_table"] = CategoryFrequencySchema(many=True).dump(data["frequency_table"])

            # Serializar la estadística en caso de ser tipo CATEGORICAL
            data["statistical_summary"] = CategoricalStatisticsSchema().dump(data["statistical_summary"])

        else:
            raise ValueError("Invalid type for univariate statistics")

        return data
    
    @post_dump
    def format_fields(self, data, **kwargs):
        data = { SchemaUtils.to_camel_case(key): SchemaUtils.recursive_camelized_element(value, **kwargs) for key, value in data.items() }
        return data
    
    
