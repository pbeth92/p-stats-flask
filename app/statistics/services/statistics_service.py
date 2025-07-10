import random
import pandas as pd
from mongoengine import ValidationError

from app.dataset.models.dataset_model import create_dataset_record_model
from app.statistics.models import Statistics

class StatisticsService:    
    def save_statistics(self, statistics: Statistics):
        try:
            statistics.save()
        except ValidationError as ve:
            raise ValueError(f"Validation error while saving statistics: {ve.messages}")
        except Exception as e:
            raise ValueError(f"Error saving statistics: {e}")
        
    def get_box_plot_data(self, dataset_id, numeric_var, category_var=None, subset_size=None):
        """
        Get data formatted for box plot visualization with optimized performance
        Args:
            dataset_id: ID of the dataset
            numeric_var: The numeric variable for the box plot
            category_var: Optional categorical variable for grouping
            subset_size: Optional size of random subset to use
        Returns:
            Dictionary with box plot data formatted for visualization
        Raises:
            ValueError: If there are issues with the data or parameters
        """
        try:
            DatasetRecord = create_dataset_record_model(dataset_id)
            
            # Verify dataset exists
            total = DatasetRecord.objects.count()
            if total == 0:
                raise ValueError(f"No records found for dataset {dataset_id}")

            # Create projection to include only required fields
            project_fields = {numeric_var: 1}
            if category_var:
                project_fields[category_var] = 1

            # Build aggregation pipeline
            pipeline = [
                {"$project": project_fields}
            ]

            # Add random sampling if subset_size is specified
            if subset_size is not None:
                if subset_size < 1:
                    raise ValueError("Subset size must be greater than 0")
                actual_size = min(subset_size, total)
                pipeline.append({"$sample": {"size": actual_size}})

            # Execute aggregation
            records = list(DatasetRecord.objects.aggregate(pipeline))
            
            if not records:
                raise ValueError("No records available for box plot")

            # Process records based on whether we have a category variable
            if category_var:
                # Group data by category
                grouped_data = {}
                for record in records:
                    cat = record.get(category_var)
                    if cat is not None:  # Skip records with null categories
                        val = record.get(numeric_var)
                        if val is not None and isinstance(val, (int, float)):  # Validate numeric values
                            if cat not in grouped_data:
                                grouped_data[cat] = []
                            grouped_data[cat].append(val)

                if not grouped_data:
                    raise ValueError(f"No valid data found for variables: {numeric_var}, {category_var}")

                data = [
                    {
                        'categorical_label': category,
                        'values': values
                    }
                    for category, values in grouped_data.items()
                ]
                
                return {
                    'data': data,
                    'data_label': numeric_var,
                    'metadata': {
                        'total_records': total,
                        'selected_records': len(records),
                        'numeric_variable': numeric_var,
                        'categorical_variable': category_var
                    }
                }
            else:
                # Process single numeric variable
                values = [
                    record.get(numeric_var)
                    for record in records
                    if record.get(numeric_var) is not None and isinstance(record.get(numeric_var), (int, float))
                ]
                
                if not values:
                    raise ValueError(f"No valid numeric values found for variable: {numeric_var}")

                return {
                    'data': [{
                        'values': values
                    }],
                    'data_label': numeric_var,
                    'metadata': {
                        'total_records': total,
                        'selected_records': len(records),
                        'numeric_variable': numeric_var
                    }
                }

        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            raise ValueError(f"Error processing box plot data: {str(e)}")