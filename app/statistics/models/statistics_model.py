import mongoengine as me

from .univariate_statistics_model import UnivariateStatistics
from .bivariate_statistics_model import BivariateStatistics

class Statistics(me.Document):
    dataset_id = me.ObjectIdField(required=True)
    univariate_statistics = me.ListField(me.EmbeddedDocumentField(UnivariateStatistics))
    bivariate_statistics = me.ListField(me.EmbeddedDocumentField(BivariateStatistics))

    meta = {
        'collection': 'statistics',
        'indexes': [
            'dataset_id'
        ]
    }