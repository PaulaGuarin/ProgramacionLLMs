import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_eliminar_outliers():
    """
    Genera un caso de prueba aleatorio para eliminar_outliers(df, columnas).
    Input : {'df': DataFrame numérico, 'columnas': lista de columnas a analizar}
    Output: (df_limpio, reporte)
    """
    n_rows    = random.randint(20, 60)
    n_cols    = random.randint(2, 4)
    col_names = [f'var_{i}' for i in range(n_cols)]
    columnas  = random.sample(col_names, k=random.randint(1, n_cols))

    data = np.random.normal(loc=0, scale=1, size=(n_rows, n_cols))

    # Inyectar outliers extremos en algunas filas
    n_outliers = random.randint(2, 5)
    for _ in range(n_outliers):
        fila = random.randint(0, n_rows - 1)
        col  = random.randint(0, n_cols - 1)
        data[fila, col] = random.choice([-1, 1]) * random.uniform(8, 15)

    df = pd.DataFrame(data, columns=col_names)

    input_data = {'df': df.copy(), 'columnas': columnas}

    # ── Ground truth ──
    mascara_outlier = pd.Series(False, index=df.index)
    reporte_filas   = []

    for col in columnas:
        serie = df[col].dropna()
        q1, q3 = np.percentile(serie, [25, 75])
        iqr     = q3 - q1
        lim_inf = q1 - 1.5 * iqr
        lim_sup = q3 + 1.5 * iqr
        es_outlier = (df[col] < lim_inf) | (df[col] > lim_sup)
        mascara_outlier |= es_outlier
        reporte_filas.append({
            'columna':             col,
            'q1':                  round(q1, 4),
            'q3':                  round(q3, 4),
            'iqr':                 round(iqr, 4),
            'limite_inf':          round(lim_inf, 4),
            'limite_sup':          round(lim_sup, 4),
            'outliers_eliminados': int(es_outlier.sum()),
        })

    df_limpio = df[~mascara_outlier].reset_index(drop=True)
    reporte   = pd.DataFrame(reporte_filas).set_index('columna')

    output_data = (df_limpio, reporte)

    return input_data, output_data


if __name__ == '__main__':
    entrada, salida_esperada = generar_caso_de_uso_eliminar_outliers()

    print("=== INPUT ===")
    print(f"Shape del DataFrame: {entrada['df'].shape}")
    print(f"Columnas a analizar: {entrada['columnas']}")
    print(entrada['df'].head())

    print("\n=== OUTPUT ESPERADO ===")
    df_limpio, reporte = salida_esperada
    print(f"Shape df_limpio: {df_limpio.shape}")
    print("\nReporte:")
    print(reporte)
