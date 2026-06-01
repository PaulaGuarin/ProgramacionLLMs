import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ── Generador corregido (reemplaza el de JuanmaLop que no tiene **kwargs) ─────

def generar_caso_de_uso_preparar_deteccion_fraude(**kwargs):
    n_rows = np.random.randint(12, 18)
    tipos = ['Retail', 'Food', 'Tech', 'Services']

    data = {
        'id_tx': range(n_rows),
        'monto': [np.random.choice([np.nan, np.random.uniform(10, 5000)])
                  for _ in range(n_rows)],
        'tipo_comercio': np.random.choice(tipos, n_rows),
    }

    df_input = pd.DataFrame(data)
    monto_max_input = float(np.random.randint(2000, 4000))

    df_step = df_input.copy()

    imputer = SimpleImputer(strategy='median')
    df_step['monto'] = imputer.fit_transform(df_step[['monto']])

    y_target = (df_step['monto'] > monto_max_input).astype(int).values

    le = LabelEncoder()
    df_step['tipo_comercio'] = le.fit_transform(df_step['tipo_comercio'])

    scaler = StandardScaler()
    df_step['monto'] = scaler.fit_transform(df_step[['monto']])

    X_res = df_step[['tipo_comercio', 'monto']].values

    input_dict = {
        'df_transacciones': df_input,
        'monto_max_normal':  monto_max_input,
    }
    return input_dict, (X_res, y_target)


# ── Solución ──────────────────────────────────────────────────────────────────

def preparar_deteccion_fraude(df_transacciones, monto_max_normal):
    """
    Prepara datos de transacciones para detección de fraude.

    Pasos:
    1. Imputa NaN en 'monto' con la mediana (SimpleImputer).
    2. Etiqueta 'sospechoso': 1 si monto imputado > monto_max_normal (antes de escalar).
    3. Codifica 'tipo_comercio' con LabelEncoder.
    4. Escala 'monto' con StandardScaler.
    5. Devuelve X=[tipo_comercio, monto] como ndarray e y=[sospechoso] como ndarray.
    """
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


if __name__ == "__main__":
    inp, (X_esp, y_esp) = generar_caso_de_uso_preparar_deteccion_fraude()
    X_r, y_r = preparar_deteccion_fraude(**inp)
    ok = np.allclose(X_r, X_esp, atol=1e-9) and np.array_equal(y_r, y_esp)
    print(f"X shape : {X_r.shape}")
    print(f"y shape : {y_r.shape}")
    print(f"Match   : {'✓' if ok else '✗'}")
