import mongoengine as me

from app.shared.shared_enums import StatisticType
class IntervalFrequencyRecord(me.EmbeddedDocument):
    interval = me.StringField(required=True)
    class_mark = me.FloatField(required=True)
    absolute_frequency = me.IntField(required=True)
    relative_frequency = me.FloatField(required=True)
    cumulative_absolute_frequency = me.IntField(required=True)
    cumulative_relative_frequency = me.FloatField(required=True)

class CategoryFrequencyRecord(me.EmbeddedDocument):
    category = me.StringField(required=True)
    absolute_frequency = me.IntField(required=True)
    relative_frequency = me.FloatField(required=True)
    cumulative_absolute_frequency = me.IntField(required=True)
    cumulative_relative_frequency = me.FloatField(required=True)

class CategoricalStatistics(me.EmbeddedDocument):
    unique_categories = me.IntField(required=True)
    most_frequent_category = me.StringField(required=True)
    frequency = me.DictField(field=me.IntField(), required=True, default=None)

class NumericalStatistics(me.EmbeddedDocument):
    mean = me.FloatField(required=True)
    median = me.FloatField(required=True)
    mode = me.FloatField(required=True)
    std = me.FloatField(required=True)
    variance = me.FloatField(required=True)
    min = me.FloatField(required=True)
    max = me.FloatField(required=True)
    percentile_25 = me.FloatField(required=True)
    percentile_75 = me.FloatField(required=True)
    skewness = me.FloatField(required=True)
    kurtosis = me.FloatField(required=True)

class UnivariateStatistics(me.EmbeddedDocument):
    variable = me.StringField(required=True)
    type = me.EnumField(enum=StatisticType, required=True)
    frequency_table = me.ListField(
        me.GenericEmbeddedDocumentField(choices=[IntervalFrequencyRecord, CategoryFrequencyRecord]),
        required=True
    )
    statistical_summary = me.GenericEmbeddedDocumentField(
        choices=[CategoricalStatistics, NumericalStatistics],
        required=True
    )
