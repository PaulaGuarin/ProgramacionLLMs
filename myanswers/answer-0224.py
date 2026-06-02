import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preparar_deteccion_fraude(df_transacciones, monto_max_normal):
    df = df_transacciones.copy()

    # 1. Imputar NaN en 'monto' con la mediana
    imputer = SimpleImputer(strategy='median')
    df['monto'] = imputer.fit_transform(df[['monto']]).ravel()

    # 2. Etiquetar ANTES de escalar
    y = (df['monto'] > monto_max_normal).astype(int).values

    # 3. Codificar 'tipo_comercio'
    le = LabelEncoder()
    df['tipo_comercio'] = le.fit_transform(df['tipo_comercio'])

    # 4. Escalar 'monto'
    scaler = StandardScaler()
    df['monto'] = scaler.fit_transform(df[['monto']]).ravel()

    # 5. X con solo tipo_comercio y monto
    X = df[['tipo_comercio', 'monto']].values

    return X, y
