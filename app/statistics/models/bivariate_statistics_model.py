import mongoengine as me

from app.shared.shared_enums import BivariateStatisticType


class NumericalCategoricalStatistic(me.EmbeddedDocument):
    anova_f_statistic = me.FloatField(required=True)
    anova_p_value = me.FloatField(required=True)
    box_plot_data = me.ListField(me.DictField(), required=True)

class CategoricalCategoricalStatistic(me.EmbeddedDocument):
    chi2 = me.FloatField(required=True)
    p_chi2 = me.FloatField(required=True)
    dof = me.IntField(required=True)
    expected = me.ListField(me.ListField(me.FloatField()), required=True)
    v_cramer = me.FloatField(required=True)
    contingency_coefficient = me.FloatField(required=True)

class NumericalNumericalStatistic(me.EmbeddedDocument):
    correlation = me.FloatField(required=True)
    p_value_correlation = me.FloatField(required=True)
    covariance = me.FloatField(required=True)

class BivariateStatistics(me.EmbeddedDocument):
    variables = me.ListField(required=True)
    type = me.EnumField(enum=BivariateStatisticType, required=True)
    contingency_table = me.ListField(me.DictField(), required=True)
    statistical_summary = me.GenericEmbeddedDocumentField(
        choices=[
            NumericalNumericalStatistic,
            CategoricalCategoricalStatistic,
            NumericalCategoricalStatistic
        ],
        required=True
    )
