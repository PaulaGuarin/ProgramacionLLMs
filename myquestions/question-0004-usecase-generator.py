import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def optimizar_modelo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
) -> dict:
    """
    Optimiza un RandomForestClassifier con GridSearchCV.

    Parámetros
    ----------
    X_train, y_train : datos de entrenamiento.
    X_test,  y_test  : datos de evaluación final.

    Retorna
    -------
    dict con:
        - mejores_params   : hiperparámetros óptimos encontrados
        - accuracy_test    : accuracy sobre el conjunto de test
        - importancia      : DataFrame con features ordenadas por importancia
    """

    # 1. Definir la cuadrícula de hiperparámetros
    param_grid = {
        "n_estimators":      [50, 100, 200],
        "max_depth":         [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    }

    # 2. Modelo base
    rf = RandomForestClassifier(random_state=42)

    # 3. GridSearchCV con validación cruzada estratificada de 5 folds
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,          # usa todos los núcleos disponibles
        verbose=1,
        refit=True,         # re-entrena con los mejores params sobre todo X_train
    )
    grid_search.fit(X_train, y_train)

    # 4. Evaluar el mejor modelo sobre test
    mejor_modelo = grid_search.best_estimator_
    y_pred       = mejor_modelo.predict(X_test)
    acc_test     = round(accuracy_score(y_test, y_pred), 4)

    # 5. Importancia de features ordenada de mayor a menor
    importancias = mejor_modelo.feature_importances_

    # Nombres de columnas si X_train es DataFrame, índices si es array
    if isinstance(X_train, pd.DataFrame):
        nombres = X_train.columns.tolist()
    else:
        nombres = [f"feature_{i}" for i in range(X_train.shape[1])]

    idx_ordenado = np.argsort(importancias)[::-1]   # orden descendente

    df_importancia = pd.DataFrame({
        "feature":    [nombres[i] for i in idx_ordenado],
        "importancia": importancias[idx_ordenado].round(4),
    })

    return {
        "mejores_params": grid_search.best_params_,
        "accuracy_test":  acc_test,
        "importancia":    df_importancia,
    }


# ── Ejemplo de uso ───────────────────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True, as_frame=True)  # as_frame=True → DataFrame

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

resultado = optimizar_modelo(X_train, y_train, X_test, y_test)

print("Mejores hiperparámetros:")
for k, v in resultado["mejores_params"].items():
    print(f"  {k}: {v}")

print(f"\nAccuracy en test: {resultado['accuracy_test']}")
print("\nImportancia de features:")
print(resultado["importancia"].to_string(index=False))