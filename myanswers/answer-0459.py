import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def generar_caso_de_uso_calcular_importancia_permutacion(**kwargs):
    n = random.randint(30, 80)
    m = random.randint(3, 6)

    X = np.random.randn(n, m)
    cols = [f"f{i}" for i in range(m)]

    y = (X[:, 0] + np.random.randn(n) > 0).astype(int)

    df = pd.DataFrame(X, columns=cols)
    target_col = "target"
    df[target_col] = y

    input_data = {
        "df": df.copy(),
        "target_col": target_col
    }

    X_df = df.drop(columns=[target_col])
    y_series = df[target_col]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_df, y_series)

    result = permutation_importance(model, X_df, y_series, n_repeats=5, random_state=42)

    output_data = {
        col: result.importances_mean[i]
        for i, col in enumerate(X_df.columns)
    }

    return input_data, output_data


def calcular_importancia_permutacion(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    result = permutation_importance(model, X, y, n_repeats=5, random_state=42)

    return {col: imp for col, imp in zip(X.columns, result.importances_mean)}
