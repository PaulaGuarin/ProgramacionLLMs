import numpy as np
import random
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def generar_caso_de_uso_evaluar_clasificador():
    """
    Genera un caso de prueba aleatorio para evaluar_clasificador(X, y, modelo).
    Input : {'X': array, 'y': array, 'modelo': clasificador sklearn}
    Output: dict con media y std de accuracy, precision y recall
    """
    n_samples  = random.randint(80, 200)
    n_features = random.randint(3, 8)
    n_classes  = random.choice([2, 3])

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features - 1),
        n_redundant=1,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random.randint(0, 999),
    )

    modelo = LogisticRegression(max_iter=500, random_state=42)

    input_data = {'X': X, 'y': y, 'modelo': modelo}

    # ── Ground truth ──
    pipeline = Pipeline(steps=[
        ('scaler',       StandardScaler()),
        ('clasificador', LogisticRegression(max_iter=500, random_state=42)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'accuracy':  make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall':    make_scorer(recall_score,    average='weighted', zero_division=0),
    }
    resultados = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)

    output_data = {
        metrica: {
            'media': round(float(np.mean(resultados[f'test_{metrica}'])), 4),
            'std':   round(float(np.std( resultados[f'test_{metrica}'])), 4),
        }
        for metrica in ['accuracy', 'precision', 'recall']
    }

    return input_data, output_data


if __name__ == '__main__':
    entrada, salida_esperada = generar_caso_de_uso_evaluar_clasificador()

    print("=== INPUT ===")
    print(f"Shape de X: {entrada['X'].shape}")
    print(f"Clases en y: {np.unique(entrada['y'])}")
    print(f"Modelo: {entrada['modelo']}")

    print("\n=== OUTPUT ESPERADO ===")
    for metrica, valores in salida_esperada.items():
        print(f"  {metrica:10s}  media={valores['media']:.4f}  std={valores['std']:.4f}")
