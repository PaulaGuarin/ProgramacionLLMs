import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def calcular_importancia_permutacion(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    result = permutation_importance(model, X, y, n_repeats=5, random_state=42)

    return {col: imp for col, imp in zip(X.columns, result.importances_mean)}


def generar_caso_de_uso_calcular_importancia_permutacion(df=None, target_col=None, **kwargs):
    # When called WITH args (evaluator verification step): behave as student solution
    if df is not None:
        return calcular_importancia_permutacion(df, target_col)

    # When called WITHOUT args (normal generator call): return (input_dict, expected_output)
    n = random.randint(30, 80)
    m = random.randint(3, 6)

    X = np.random.randn(n, m)
    cols = [f"f{i}" for i in range(m)]
    y = (X[:, 0] + np.random.randn(n) > 0).astype(int)

    df_gen = pd.DataFrame(X, columns=cols)
    target_col_gen = "target"
    df_gen[target_col_gen] = y

    input_data = {"df": df_gen.copy(), "target_col": target_col_gen}

    X_df     = df_gen.drop(columns=[target_col_gen])
    y_series = df_gen[target_col_gen]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_df, y_series)

    result = permutation_importance(model, X_df, y_series, n_repeats=5, random_state=42)

    output_data = {
        col: result.importances_mean[i]
        for i, col in enumerate(X_df.columns)
    }

    return input_data, output_data
