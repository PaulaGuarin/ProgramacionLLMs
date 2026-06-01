import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def calcular_importancia_permutacion(df, target_col):
    """
    Calcula la importancia de variables por permutación usando RandomForest.

    Pasos:
    1. Separa X (features) de y (target).
    2. Entrena RandomForestClassifier con random_state=42.
    3. Calcula permutation_importance con n_repeats=5, random_state=42.
    4. Devuelve dict {nombre_columna: importancia_promedio}.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    result = permutation_importance(model, X, y, n_repeats=5, random_state=42)

    return {col: imp for col, imp in zip(X.columns, result.importances_mean)}


if __name__ == "__main__":
    import random
    random.seed(7)
    np.random.seed(7)

    n = 60
    X_demo = np.random.randn(n, 4)
    df_demo = pd.DataFrame(X_demo, columns=['edad', 'ingreso', 'score', 'deuda'])
    df_demo['target'] = (X_demo[:, 0] + np.random.randn(n) > 0).astype(int)

    resultado = calcular_importancia_permutacion(df_demo, 'target')
    print("Importancias por permutación:")
    for col, imp in sorted(resultado.items(), key=lambda x: -x[1]):
        bar = "█" * int(max(0, imp) * 100)
        print(f"  {col:<15} {imp:+.6f}  {bar}")
