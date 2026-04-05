import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier


def evaluar_clasificador(X, y, modelo):
    pipeline = Pipeline(steps=[
        ("scaler",       StandardScaler()),
        ("clasificador", modelo),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ← zero_division=0 evita el warning cuando una clase no tiene predicciones
    scoring = {
        "accuracy":  make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average="weighted", zero_division=0),
        "recall":    make_scorer(recall_score,    average="weighted", zero_division=0),
    }

    resultados = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    resumen = {}
    for metrica in ["accuracy", "precision", "recall"]:
        scores = resultados[f"test_{metrica}"]
        resumen[metrica] = {
            "media": round(float(np.mean(scores)), 4),
            "std":   round(float(np.std(scores)),  4),
        }
    return resumen


# ── Ejemplo de uso ───────────────────────────────────────────────────────────

X, y = load_iris(return_X_y=True)

for nombre, modelo in [
    ("Regresión Logística", LogisticRegression(max_iter=200)),
    ("Random Forest",       RandomForestClassifier(n_estimators=100, random_state=42)),
    ("Dummy (baseline)",    DummyClassifier(strategy="most_frequent")),
]:
    resultado = evaluar_clasificador(X, y, modelo)
    print(f"\n{nombre}")
    for metrica, valores in resultado.items():
        print(f"  {metrica:10s}  media={valores['media']:.4f}  std={valores['std']:.4f}")