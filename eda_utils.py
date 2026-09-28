"""
Reusable EDA helper functions for the Machine Learning course project.

Keep dataset-specific cleaning, assumptions, and interpretation in the notebook.
"""

from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path):
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def basic_overview(df):
    """Display dataset size, duplicate count, data types, and first rows."""
    print("Shape:", df.shape)
    print("Duplicated rows:", df.duplicated().sum())

    print("\nData types:")
    display(df.dtypes.to_frame("dtype"))

    print("\nFirst rows:")
    display(df.head())


def missing_values_summary(df):
    """Return columns with missing values, sorted by missing percentage."""
    missing = pd.DataFrame({
        "missing_n": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100
    })

    missing = missing[missing["missing_n"] > 0]
    missing = missing.sort_values("missing_pct", ascending=False)

    return missing


def numeric_summary(df, numeric_cols):
    """Return descriptive statistics for numeric columns."""
    if not numeric_cols:
        return pd.DataFrame()

    return df[numeric_cols].describe().T


def get_column_types(df):
    """Return lists of columns detected as numeric and categorical."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return numeric_cols, categorical_cols


def categorical_summary(df, categorical_cols):
    """Return cardinality, missingness, and most common value for categoricals."""
    summary = []

    for col in categorical_cols:
        counts = df[col].value_counts(dropna=False)

        summary.append({
            "variable": col,
            "unique_values": df[col].nunique(dropna=True),
            "missing_n": df[col].isna().sum(),
            "most_common": counts.index[0] if len(counts) > 0 else None,
            "most_common_n": counts.iloc[0] if len(counts) > 0 else None
        })

    return pd.DataFrame(summary)


def clean_string_columns(df):
    """
    Strip leading/trailing whitespace from text-like columns while preserving
    actual missing values instead of converting them to the string 'nan'.
    """
    string_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in string_cols:
        df[col] = df[col].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    return df


def plot_numeric_distributions(df, numeric_cols):
    """Plot a histogram and KDE for each numeric column."""
    for col in numeric_cols:
        plt.figure(figsize=(7, 4))

        sns.histplot(
            data=df,
            x=col,
            kde=True
        )

        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()


def plot_categorical_distributions(df, categorical_cols, max_categories=15):
    """Plot categorical distributions when cardinality is manageable."""
    for col in categorical_cols:
        counts = df[col].value_counts(dropna=False)

        if len(counts) <= max_categories:
            plt.figure(figsize=(8, 4))

            sns.countplot(
                data=df,
                x=col,
                order=df[col].value_counts().index
            )

            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        else:
            print(
                f"{col}: {len(counts)} categories. "
                "Plot skipped because cardinality is too high."
            )


def plot_numeric_vs_target(df, numeric_cols, target):
    """Plot each numeric predictor against a continuous target."""
    for col in numeric_cols:
        if col == target:
            continue

        plt.figure(figsize=(7, 4))

        sns.scatterplot(
            data=df,
            x=col,
            y=target,
            alpha=0.6
        )

        plt.title(f"{col} vs {target}")
        plt.tight_layout()
        plt.show()


def plot_categorical_vs_target(df, categorical_cols, target, max_categories=15):
    """Plot target distributions across low-cardinality categorical variables."""
    for col in categorical_cols:
        if df[col].nunique(dropna=False) <= max_categories:
            plt.figure(figsize=(8, 4))

            sns.boxplot(
                data=df,
                x=col,
                y=target
            )

            plt.title(f"{target} by {col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


def target_correlations(df, numeric_cols, target):
    """Return Pearson and Spearman correlations between predictors and target."""
    cols = [col for col in numeric_cols if col != target]

    pearson = (
        df[cols + [target]]
        .corr(method="pearson")[target]
        .drop(target)
    )

    spearman = (
        df[cols + [target]]
        .corr(method="spearman")[target]
        .drop(target)
    )

    correlations = pd.DataFrame({
        "pearson": pearson,
        "spearman": spearman
    })

    correlations["abs_pearson"] = correlations["pearson"].abs()

    return correlations.sort_values(
        "abs_pearson",
        ascending=False
    )


def categorical_target_summary(df, categorical_cols, target):
    """Return count, mean, median, and SD of target for each category."""
    summaries = {}

    for col in categorical_cols:
        summaries[col] = (
            df.groupby(col, observed=True)[target]
            .agg(["count", "mean", "median", "std"])
            .sort_values("mean", ascending=False)
        )

    return summaries


def plot_correlation_heatmap(df, numeric_cols):
    """Plot and return a Pearson correlation matrix for numeric columns."""
    corr = df[numeric_cols].corr(method="pearson")

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.show()

    return corr


def high_correlation_pairs(df, numeric_cols, threshold=0.7):
    """Return numeric-variable pairs whose absolute correlation meets threshold."""
    corr = df[numeric_cols].corr().abs()

    pairs = []

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            value = corr.iloc[i, j]

            if value >= threshold:
                pairs.append({
                    "variable_1": corr.columns[i],
                    "variable_2": corr.columns[j],
                    "correlation": value
                })

    if not pairs:
        return pd.DataFrame(
            columns=["variable_1", "variable_2", "correlation"]
        )

    return pd.DataFrame(pairs).sort_values(
        "correlation",
        ascending=False
    )


def plot_target_over_time(df, date_col, target):
    """Plot mean target value by date."""
    daily_target = (
        df.groupby(date_col)[target]
        .mean()
        .sort_index()
    )

    plt.figure(figsize=(10, 4))
    daily_target.plot()

    plt.title(f"Mean {target} over time")
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
