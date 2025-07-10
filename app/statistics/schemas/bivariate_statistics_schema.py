from marshmallow import Schema, fields, pre_dump, post_dump

from app.shared.schema_utils import SchemaUtils
from app.shared.shared_enums import BivariateStatisticType

class NumericalNumericalStatisticSchema(Schema):
    correlation = fields.Float(required=True)
    p_value_correlation = fields.Float(required=True)
    covariance = fields.Float(required=True)

class CategoricalCategoricalStatisticSchema(Schema):
    chi2 = fields.Float(required=True)
    p_chi2 = fields.Float(required=True)
    dof = fields.Int(required=True)
    expected = fields.List(fields.List(fields.Float()), required=True)
    v_cramer = fields.Float(required=True)
    contingency_coefficient = fields.Float(required=True)

class NumericalCategoricalStatisticSchema(Schema):
    anova_f_statistic = fields.Float(required=True)
    anova_p_value = fields.Float(required=True)
    box_plot_data = fields.List(fields.Dict(), required=True)

class BivariateStatisticsSchema(Schema):
    variables = fields.List(fields.Str(), required=True)
    type = fields.Enum(
        enum=BivariateStatisticType,
        by_value=True,
        required=True
    )
    contingency_table = fields.List(fields.Dict(), required=True)
    statistical_summary = fields.Raw(required=True)

    @pre_dump
    def validate_and_flatten(self, data, **kwargs):
        """This method ensures that statistical_summary has the correct schema
        depending on the type (NUMERICAL_NUMERICAL, CATEGORICAL_CATEGORICAL, or NUMERICAL_CATEGORICAL)"""
        
        if data["type"] == BivariateStatisticType.NUMERICAL_NUMERICAL:
            data["statistical_summary"] = NumericalNumericalStatisticSchema().dump(data["statistical_summary"])
        elif data["type"] == BivariateStatisticType.CATEGORICAL_CATEGORICAL:
            data["statistical_summary"] = CategoricalCategoricalStatisticSchema().dump(data["statistical_summary"])
        elif data["type"] == BivariateStatisticType.CATEGORICAL_NUMERICAL:
            data["statistical_summary"] = NumericalCategoricalStatisticSchema().dump(data["statistical_summary"])
        else:
            raise ValueError("Invalid type for bivariate statistics")

        return data

    @post_dump
    def format_fields(self, data, **kwargs):
        data = { SchemaUtils.to_camel_case(key): SchemaUtils.recursive_camelized_element(value, **kwargs) for key, value in data.items() }
        return data