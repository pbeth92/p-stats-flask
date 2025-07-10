import mongoengine as me

from app.dataset.utils.utils import DatasetStatus

class Variable(me.EmbeddedDocument):
    name = me.StringField(required=True)
    type = me.StringField(required=True)

class Dataset(me.Document):
    name = me.StringField(required=True)
    description = me.StringField(required=True)
    tags = me.ListField(me.StringField(), default=list)
    n_cols = me.IntField(required=True)
    n_rows = me.IntField(required=True)
    size = me.FloatField(required=True)
    variables = me.ListField(me.EmbeddedDocumentField(Variable), required=True)
    create_at = me.DateTimeField(required=True)
    update_at = me.DateTimeField(required=True)
    status = me.EnumField(
        enum=DatasetStatus,
        required=True,
        default=DatasetStatus.PROCESSING
    )

    meta = {
        'collection': 'dataset'
    }
    
def create_dataset_record_model(dataset_id: str):
    try:
        class DatasetRecord(me.DynamicDocument):

            meta = {
                'collection': f'records_{dataset_id}'
            }

        print(f"DatasetRecord model created for dataset {dataset_id}")
        return DatasetRecord
    
    except Exception as e:
        print(f"Error creating model: {e}")
        return None