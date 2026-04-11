import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

def generar_caso_de_uso_optimizar_modelo():
    """
    Genera un caso de prueba aleatorio para
    optimizar_modelo(X_train, y_train, X_test, y_test).
    Input : {'X_train', 'y_train', 'X_test', 'y_test'}
    Output: dict con mejores_params, accuracy_test e importancia
    """
    n_samples  = random.randint(100, 300)
    n_features = random.randint(4, 10)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features - 2),
        n_redundant=2,
        random_state=random.randint(0, 999),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    input_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test':  X_test,
        'y_test':  y_test,
    }

    # ── Ground truth ──
    param_grid = {
        'n_estimators':      [50, 100, 200],
        'max_depth':         [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
    }
    rf = RandomForestClassifier(random_state=42)
    gs = GridSearchCV(rf, param_grid, scoring='accuracy', cv=5, n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)

    mejor_modelo = gs.best_estimator_
    y_pred       = mejor_modelo.predict(X_test)
    acc_test     = round(accuracy_score(y_test, y_pred), 4)
    importancias = mejor_modelo.feature_importances_
    idx_ord      = np.argsort(importancias)[::-1]

    df_importancia = pd.DataFrame({
        'feature':    [f'feature_{i}' for i in idx_ord],
        'importancia': importancias[idx_ord].round(4),
    })

    output_data = {
        'mejores_params': gs.best_params_,
        'accuracy_test':  acc_test,
        'importancia':    df_importancia,
    }

    return input_data, output_data


if __name__ == '__main__':
    entrada, salida_esperada = generar_caso_de_uso_optimizar_modelo()

    print("=== INPUT ===")
    print(f"X_train shape: {entrada['X_train'].shape}")
    print(f"X_test shape:  {entrada['X_test'].shape}")

    print("\n=== OUTPUT ESPERADO ===")
    print(f"Mejores parámetros: {salida_esperada['mejores_params']}")
    print(f"Accuracy en test:   {salida_esperada['accuracy_test']}")
    print("\nImportancia de features:")
    print(salida_esperada['importancia'])
