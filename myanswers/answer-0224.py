import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def preparar_deteccion_fraude(df_transacciones, monto_max_normal):
    """
    Prepara los datos de transacciones para detección de fraude.

    Pasos:
    1. Imputa NaN en 'monto' con la mediana (SimpleImputer).
    2. Etiqueta 'sospechoso': 1 si monto (imputado, antes de escalar) > monto_max_normal.
    3. Codifica 'tipo_comercio' con LabelEncoder.
    4. Escala 'monto' con StandardScaler.
    5. Devuelve X (tipo_comercio + monto) como ndarray e y (sospechoso) como ndarray.
    """
    df = df_transacciones.copy()

    # 1. Imputar NaN en 'monto' con la mediana
    imputer = SimpleImputer(strategy='median')
    df['monto'] = imputer.fit_transform(df[['monto']]).ravel()

    # 2. Etiquetar ANTES de escalar (monto imputado, no escalado)
    y = (df['monto'] > monto_max_normal).astype(int).to_numpy()

    # 3. Codificar 'tipo_comercio' con LabelEncoder
    le = LabelEncoder()
    df['tipo_comercio'] = le.fit_transform(df['tipo_comercio'])

    # 4. Escalar 'monto' con StandardScaler
    scaler = StandardScaler()
    df['monto'] = scaler.fit_transform(df[['monto']]).ravel()

    # 5. Construir X con las dos columnas procesadas
    X = df[['tipo_comercio', 'monto']].to_numpy()

    return X, y


if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)

    # Datos de prueba rápida
    df_test = pd.DataFrame({
        'tipo_comercio': np.random.choice(['Retail', 'Food', 'Tech', 'Travel'], 20),
        'monto': [np.nan if i % 7 == 0 else round(np.random.lognormal(4, 1), 2)
                  for i in range(20)],
    })
    monto_max = 80.0

    X, y = preparar_deteccion_fraude(df_test, monto_max)
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")
    print(f"X dtype : {X.dtype}")
    print(f"Sospechosos: {y.sum()} de {len(y)}")
    print(f"X media monto (col 1) ≈ 0: {X[:, 1].mean():.4f}")
    print(f"X std  monto (col 1) ≈ 1: {X[:, 1].std():.4f}")
