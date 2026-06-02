import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def calcular_importancia_permutacion(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    result = permutation_importance(model, X, y, n_repeats=5, random_state=42)

    return {col: imp for col, imp in zip(X.columns, result.importances_mean)}
