# chi_square_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway

def anova_test_numerical_features(df, numeric_columns, target_column='churn', plot=True):
    '''
    Perform ANOVA test for numerical columns against a binary target variable and plot histograms.
    Converts target to binary 0/1 if it's not numeric.

    Parameters:
    - df: pandas DataFrame
    - numeric_columns: list of numerical column names to test
    - target_column: binary target variable (e.g., 'churn')
    - plot: whether to show distribution histograms

    Returns:
    - results_df: DataFrame with p-values and significance flags
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import f_oneway

    results = []

    # Convert target to binary if not numeric
    if df[target_column].dtype != 'int64' and df[target_column].dtype != 'float64':
        if set(df[target_column].unique()) == {'yes', 'no'}:
            df[target_column] = df[target_column].map({'no': 0, 'yes': 1})
        else:
            raise ValueError(f"Target column '{target_column}' must be binary and convertible to 0/1.")

    for col in numeric_columns:
        if col not in df.columns or df[col].nunique() <= 1:
            continue
        try:
            group1 = df[df[target_column] == 0][col].dropna()
            group2 = df[df[target_column] == 1][col].dropna()
            f_stat, p_value = f_oneway(group1, group2)
            significant = p_value < 0.05
            results.append({
                'feature': col,
                'p_value': round(p_value, 4),
                'significant': significant
            })

            if plot:
                plt.figure(figsize=(8, 4))
                sns.histplot(data=df, x=col, hue=target_column, kde=True, element='step')
                plt.title(f'{col} Distribution by {target_column} (p={p_value:.4f})')
                plt.tight_layout()
                plt.show()

        except Exception as e:
            results.append({
                'feature': col,
                'p_value': None,
                'significant': False,
                'error': str(e)
            })

    return pd.DataFrame(results)



def chi_square_test(df, cat_column, target_column='churn', plot=True):
    '''
    Perform Chi-Square Test of Independence between a categorical feature and the target (e.g., churn).
    
    Parameters:
    - df (pd.DataFrame): The dataset
    - cat_column (str): The name of the categorical feature to test
    - target_column (str): The name of the binary target column (default is 'churn')
    - plot (bool): If True, show a countplot of the feature grouped by target

    Returns:
    - p_value (float): The p-value from the Chi-Square test
    - conclusion (str): Whether the result is statistically significant
    '''
    # Create contingency table
    contingency_table = pd.crosstab(df[cat_column], df[target_column])

    # Perform Chi-Square Test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Visualize
    if plot:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=cat_column, hue=target_column)
        plt.title(f'Distribution of {cat_column} by {target_column}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Interpret result
    significance = 0.05
    if p < significance:
        conclusion = f"✅ Significant relationship (p-value = {p:.4f})"
    else:
        conclusion = f"❌ No significant relationship (p-value = {p:.4f})"
    
    return p, conclusion

def chi_square_test_batch(df, cat_columns, target_column='churn', plot=False):
    """
    Run chi-square test on multiple categorical columns against a binary target.

    Parameters:
    - df: DataFrame
    - cat_columns: list of categorical column names
    - target_column: name of the binary target
    - plot: whether to plot each distribution

    Returns:
    - DataFrame summarizing p-values and test significance
    """
    results = []

    for col in cat_columns:
        try:
            p, conclusion = chi_square_test(df, col, target_column, plot=plot)
            results.append({
                'feature': col,
                'p_value': round(p, 4),
                'significant': p < 0.05
            })
        except Exception as e:
            results.append({
                'feature': col,
                'p_value': None,
                'significant': False,
                'error': str(e)
            })

    return pd.DataFrame(results)


def plot_significant_categorical_proportions(df, significant_features, target_column='churn'):
    '''
    Plots proportion bar charts of significant categorical features vs the target (e.g., churn),
    with data labels on each bar segment.

    Parameters:
    - df: pandas DataFrame
    - significant_features: list of features to plot
    - target_column: the target variable to compare against
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt

    for col in significant_features:
        if col not in df.columns:
            continue

        prop_df = pd.crosstab(df[col], df[target_column], normalize='index')
        ax = prop_df.plot(kind='bar', stacked=True, colormap='Set2', figsize=(8, 4))

        # Add data labels
        for p in ax.patches:
            height = p.get_height()
            if height > 0.01:  # Only show labels for visible segments
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    p.get_y() + height / 2,
                    f'{height:.1%}',
                    ha='center', va='center',
                    fontsize=9
                )

        plt.title(f'Proportion of {target_column} by {col}')
        plt.ylabel('Proportion')
        plt.xticks(rotation=45)
        plt.legend(title=target_column)
        plt.tight_layout()
        plt.show()



