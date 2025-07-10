from typing import List
from bson import ObjectId
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pearsonr, f_oneway

from app.statistics.models import Statistics
from app.shared.shared_enums import BivariateStatisticType
from app.statistics.models.bivariate_statistics_model import BivariateStatistics, CategoricalCategoricalStatistic, NumericalCategoricalStatistic, NumericalNumericalStatistic

class BivariateStatisticsService:
    MAX_VARIABLES = 50
    
    def get_bivariate_statistics(self, dataset_id: ObjectId) -> List[BivariateStatistics]:
        try:
            statistics = Statistics.objects(dataset_id=dataset_id).only("bivariate_statistics").first()
            
            return statistics.bivariate_statistics

        except Exception as e:
            raise ValueError(f"Failed getting bivariate statistics from database: {e}")

    def generate_bivariate_statistics(self, df) -> List[BivariateStatistics]:
        # Limit to first MAX_VARIABLES columns if DataFrame has more variables
        if len(df.columns) > self.MAX_VARIABLES:
            df = df.iloc[:, :self.MAX_VARIABLES]
        
        bivariate_statistics = []

        numerical_vars = []
        categorical_vars = []

        for column in df.columns:
            column_data = df[column]

            if pd.api.types.is_numeric_dtype(column_data):
                numerical_vars.append(column)
            elif column_data.dtype == 'object':
                categorical_vars.append(column)

        
        if len(numerical_vars) > 1:
            bivariate_statistics.extend(self._numerical_numerical_analysis(df, numerical_vars))
        if len(categorical_vars) > 1:
           bivariate_statistics.extend(self._categorical_categorical_analysis(df, categorical_vars))
        if len(numerical_vars) > 0 and len(categorical_vars) > 0:
            bivariate_statistics.extend(self._numerical_categorical_analysis(df,categorical_vars,  numerical_vars))

        return bivariate_statistics
    
    @staticmethod
    def _categorical_categorical_analysis(df, categorical_vars):
        results = []

        for i, var1 in enumerate(categorical_vars):
            for var2 in categorical_vars[i+1:]:

                contingency_table = pd.crosstab(df[var1], df[var2], margins=True, margins_name="Total")

                contingency_table_list = [
                    {"label": row_label, "values": [{k: v} for k, v in row_data.items()]}
                    for row_label, row_data in contingency_table.iterrows()
                ]

                # Realizar la prueba Chi-Cuadrado de Independencia
                chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)

                # Convertir expected a una lista de listas para hacerla serializable
                expected_list = expected.tolist()

                # Calcular el Coeficiente de Contingencia de Pearson (C)
                n = contingency_table.sum().sum()
                contingency_coefficient = np.sqrt(chi2 / (chi2 + n)) if n > 0 else np.nan
                
                # Calcular el V de Cramer
                v_cramer = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1))) if n > 0 else np.nan

                results.append(
                    BivariateStatistics(
                        variables=[var1, var2],
                        type=BivariateStatisticType.CATEGORICAL_CATEGORICAL.value,
                        contingency_table=contingency_table_list,
                        statistical_summary=CategoricalCategoricalStatistic(
                            chi2=chi2,
                            p_chi2=p_chi2,
                            dof=dof,
                            expected=expected_list,
                            v_cramer=v_cramer,
                            contingency_coefficient=contingency_coefficient
                        )
                    )   
                )

        return results
    
    @staticmethod
    def _numerical_numerical_analysis(df, numerical_vars):
        results = []

        for i, var1 in enumerate(numerical_vars):
            for var2 in numerical_vars[i+1:]:
                # Obtener los datos de las dos variables
                col_data1 = df[var1]
                col_data2 = df[var2]

                # Número de clases (bins) con el método de Sturges
                num_classes1 = int(1 + 3.3 * np.log10(len(col_data1)))
                num_classes2 = int(1 + 3.3 * np.log10(len(col_data2)))

                # Definir los intervalos de clase usando el rango de los datos
                min_value1 = col_data1.min()
                max_value1 = col_data1.max()
                min_value2 = col_data2.min()
                max_value2 = col_data2.max()

                bins1 = np.linspace(min_value1, max_value1, num_classes1 + 1)  # Intervalos para var1
                bins2 = np.linspace(min_value2, max_value2, num_classes2 + 1)  # Intervalos para var2

                # Etiquetas de los intervalos
                labels1 = [f"{round(bins1[i], 2)} - {round(bins1[i+1], 2)}" for i in range(num_classes1)]
                labels2 = [f"{round(bins2[i], 2)} - {round(bins2[i+1], 2)}" for i in range(num_classes2)]

                # Discretizar los datos de las dos variables
                binned_data1 = pd.cut(col_data1, bins=bins1, labels=labels1, include_lowest=True)
                binned_data2 = pd.cut(col_data2, bins=bins2, labels=labels2, include_lowest=True)

                # Crear la tabla de contingencia entre los intervalos de las dos variables
                contingency_table = pd.crosstab(binned_data1, binned_data2, margins=True, margins_name="Total")

                contingency_table_list = [
                    {"label": row_label, "values": [{k: v} for k, v in row_data.items()]}
                    for row_label, row_data in contingency_table.iterrows()
                ]

                # Calcular el Coeficiente de Correlación de Pearson entre las variables originales
                correlation, p_value_correlation = pearsonr(col_data1, col_data2)

                # Covarianza
                covariance = np.cov(col_data1, col_data2, ddof=0)[0, 1]

                results.append(
                    BivariateStatistics(
                        variables=[var1, var2],
                        type=BivariateStatisticType.NUMERICAL_NUMERICAL,
                        contingency_table=contingency_table_list,
                        statistical_summary=NumericalNumericalStatistic(
                            correlation=correlation,
                            p_value_correlation=p_value_correlation,
                            covariance=covariance
                        )
                    )
                )

        return results
    
    @staticmethod
    def _numerical_categorical_analysis(df, categorical_vars, numerical_vars):
        results = []

        for var1 in  categorical_vars:
            for var2 in numerical_vars:
                if df[var1].nunique() < 2:
                    continue

                # Obtener los datos de las dos variables
                col_data_qualitative = df[var1]  # Variable cualitativa
                col_data_quantitative = df[var2]  # Variable cuantitativa

                # Número de clases (bins) para la variable cuantitativa
                num_classes = int(1 + 3.3 * np.log10(len(col_data_quantitative)))

                # Definir los intervalos de clase para la variable cuantitativa
                min_value = col_data_quantitative.min()
                max_value = col_data_quantitative.max()
                bins = np.linspace(min_value, max_value, num_classes + 1)  # Genera los intervalos
                labels = [f"{round(bins[i], 2)} - {round(bins[i+1], 2)}" for i in range(num_classes)]

                # Discretizar los datos de la variable cuantitativa
                binned_data = pd.cut(col_data_quantitative, bins=bins, labels=labels, include_lowest=True)

                # Crear la tabla de contingencia entre los intervalos de la variable cuantitativa y la variable cualitativa
                contingency_table = pd.crosstab(col_data_qualitative, binned_data, margins=True, margins_name="Total")

                contingency_table_list = [
                    {"label": row_label, "values": [{k: v} for k, v in row_data.items()]}
                    for row_label, row_data in contingency_table.iterrows()
                ]

                # Realizar un ANOVA para comparar las medias de la variable cuantitativa entre los grupos cualitativos
                unique_groups = col_data_qualitative.unique()
                group_data = [col_data_quantitative[col_data_qualitative == group] for group in unique_groups]
                # Comprobar si algún grupo tiene varianza cero (todos los valores son iguales)
                # o si tiene menos de dos muestras, lo que también invalida el test.
                is_constant = any(group.nunique() < 2 or len(group) < 2 for group in group_data)

                if is_constant:
                    # Si algún grupo es constante, el F-statistic no es definible.
                    # Asignamos NaN (Not a Number) para indicar que el test no se pudo realizar.
                    f_stat, p_anova = np.nan, np.nan
                else:
                    # Si todos los grupos tienen varianza, procedemos con el cálculo.
                    f_stat, p_anova = f_oneway(*group_data)

                # Crear datos optimizados para el gráfico de caja y bigote (Box Plot)
                box_plot_data = [
                    {
                        "category": unique_groups[i],
                        "values": [
                            float(group.min()),  
                            float(group.quantile(0.25)),
                            float(group.median()),
                            float(group.quantile(0.75)),
                            float(group.max())
                        ]
                    }
                    for i, group in enumerate(group_data)
                ]

                results.append(
                    BivariateStatistics(
                        variables=[var1, var2],
                        type=BivariateStatisticType.CATEGORICAL_NUMERICAL,
                        contingency_table=contingency_table_list,
                        statistical_summary=NumericalCategoricalStatistic(
                            box_plot_data=box_plot_data,
                            anova_f_statistic=f_stat,
                            anova_p_value=p_anova           
                        )
                    )
                )

        return results