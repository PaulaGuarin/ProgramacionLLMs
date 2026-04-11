import pandas as pd
import numpy as np

def eliminar_outliers(
    df: pd.DataFrame,
    columnas: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = df.copy()
    mascara_outlier = pd.Series(False, index=df.index)
    reporte_filas = []

    for col in columnas:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no existe en el DataFrame.")

        serie = df[col].dropna()

        q1, q3 = np.percentile(serie, [25, 75])
        iqr = q3 - q1

        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr

        es_outlier = (df[col] < limite_inf) | (df[col] > limite_sup)
        n_outliers = int(es_outlier.sum())

        mascara_outlier |= es_outlier

        reporte_filas.append({
            "columna":             col,
            "q1":                  round(q1, 4),
            "q3":                  round(q3, 4),
            "iqr":                 round(iqr, 4),
            "limite_inf":          round(limite_inf, 4),
            "limite_sup":          round(limite_sup, 4),
            "outliers_eliminados": n_outliers,
        })

    df_limpio = df[~mascara_outlier].reset_index(drop=True)
    reporte   = pd.DataFrame(reporte_filas).set_index("columna")

    return df_limpio, reporte


# ── Ejemplo de uso ──────────────────────────────────────────────────────────
np.random.seed(42)

df = pd.DataFrame({
    "edad":    np.concatenate([np.random.normal(35, 5, 97), [120, -10, 200]]),
    "salario": np.concatenate([np.random.normal(3000, 400, 97), [50000, -500, 40000]]),
    "hijos":   np.random.randint(0, 5, 100),
})

df_limpio, reporte = eliminar_outliers(df, columnas=["edad", "salario"])

print(f"Filas originales : {len(df)}")
print(f"Filas conservadas: {len(df_limpio)}")
print()
print(reporte)
