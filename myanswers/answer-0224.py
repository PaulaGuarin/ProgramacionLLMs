import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import random


def generar_caso_de_uso_preparar_deteccion_fraude():
    """
    Genera un caso de uso aleatorio para preparar_deteccion_fraude.

    Variaciones cubiertas:
    - n_rows           : 20 – 80 filas.
    - tipos_comercio   : 3 – 6 categorías del pool.
    - NaN en 'monto'   : 1 – 15 % de filas.
    - Montos           : distribución log-normal (cola larga realista).
    - monto_max_normal : percentil p30–p75 de los montos → proporción sospechosos variable.
    - Caso edge        : 25 % de prob. de un monto extremo (10× la media).

    Retorna
    -------
    input_dict   : dict con claves 'df_transacciones' y 'monto_max_normal'
    (X_esp, y_esp) : tupla de numpy arrays (ground truth)
    """

    n_rows = random.randint(20, 80)

    pool_tipos = ['Retail', 'Food', 'Tech', 'Travel', 'Entertainment',
                  'Health', 'Education', 'Finance']
    tipos_usados = random.sample(pool_tipos, random.randint(3, 6))

    # Montos log-normales
    media_log = random.uniform(3.5, 6.0)
    sigma_log  = random.uniform(0.4, 1.0)
    montos_raw = np.round(np.random.lognormal(mean=media_log, sigma=sigma_log, size=n_rows), 2)

    # Caso edge: monto extremo
    if random.random() < 0.25:
        montos_raw[random.randint(0, n_rows - 1)] = round(montos_raw.mean() * 10, 2)

    # Inyectar NaN
    n_nans  = max(1, int(n_rows * random.uniform(0.01, 0.15)))
    nan_idx = random.sample(range(n_rows), n_nans)
    montos_con_nan = montos_raw.astype(float).copy()
    montos_con_nan[nan_idx] = np.nan

    # monto_max_normal desde percentil aleatorio
    percentil = random.uniform(30, 75)
    monto_max_normal = round(float(np.percentile(montos_raw, percentil)), 2)

    df = pd.DataFrame({
        'tipo_comercio': np.random.choice(tipos_usados, n_rows),
        'monto':         montos_con_nan,
    })

    # Ground truth
    df_step = df.copy()
    imputer = SimpleImputer(strategy='median')
    df_step['monto'] = imputer.fit_transform(df_step[['monto']]).ravel()
    y_esp = (df_step['monto'] > monto_max_normal).astype(int).to_numpy()
    le = LabelEncoder()
    df_step['tipo_comercio'] = le.fit_transform(df_step['tipo_comercio'])
    scaler = StandardScaler()
    df_step['monto'] = scaler.fit_transform(df_step[['monto']]).ravel()
    X_esp = df_step[['tipo_comercio', 'monto']].to_numpy()

    input_dict = {
        'df_transacciones': df.copy(),
        'monto_max_normal':  monto_max_normal,
    }
    return input_dict, (X_esp, y_esp)


if __name__ == "__main__":
    entrada, (X_esp, y_esp) = generar_caso_de_uso_preparar_deteccion_fraude()
    df_in = entrada['df_transacciones']
    print("=== INPUT ===")
    print(f"monto_max_normal : {entrada['monto_max_normal']}")
    print(f"Shape            : {df_in.shape}")
    print(f"NaN en monto     : {df_in['monto'].isna().sum()}")
    print(f"tipos_comercio   : {sorted(df_in['tipo_comercio'].unique())}")
    print(df_in.head(8).to_string())
    print("\n=== OUTPUT ESPERADO ===")
    print(f"X shape: {X_esp.shape}  |  y shape: {y_esp.shape}")
    print(f"X (primeras 8 filas):\n{X_esp[:8]}")
    print(f"y (primeros 8 valores): {y_esp[:8]}")
    print(f"Proporción sospechosos: {y_esp.mean():.1%}")
