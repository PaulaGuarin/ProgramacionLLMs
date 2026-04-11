import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_analizar_ventas():
    """
    Genera un caso de prueba aleatorio para la función analizar_ventas(df).
    Input : {'df': DataFrame con columnas fecha, producto, cantidad, precio_unitario}
    Output: DataFrame agrupado por (mes, producto) con ingreso_total y cantidad_promedio
    """
    n_rows    = random.randint(10, 30)
    productos = random.sample(['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Auriculares', 'Webcam'], k=random.randint(2, 4))
    meses     = ['2024-01', '2024-02', '2024-03', '2024-04']

    fechas     = [f"{random.choice(meses)}-{random.randint(1,28):02d}" for _ in range(n_rows)]
    productos_ = [random.choice(productos) for _ in range(n_rows)]
    cantidades = np.random.randint(1, 20, size=n_rows).astype(float)
    precios    = np.round(np.random.uniform(10, 2000, size=n_rows), 2)

    # Introducir ~10% de NaNs
    for arr in [cantidades, precios]:
        mask = np.random.choice([True, False], size=n_rows, p=[0.1, 0.9])
        arr[mask] = np.nan

    df = pd.DataFrame({
        'fecha':           fechas,
        'producto':        productos_,
        'cantidad':        cantidades,
        'precio_unitario': precios,
    })

    input_data = {'df': df.copy()}

    # ── Ground truth ──
    df_gt = df.copy()
    df_gt.dropna(subset=['fecha', 'producto', 'cantidad', 'precio_unitario'], inplace=True)
    df_gt['fecha']         = pd.to_datetime(df_gt['fecha'])
    df_gt['mes']           = df_gt['fecha'].dt.to_period('M')
    df_gt['ingreso_total'] = np.multiply(df_gt['cantidad'], df_gt['precio_unitario'])

    output_data = (
        df_gt.groupby(['mes', 'producto'], sort=True)
        .agg(
            ingreso_total     = ('ingreso_total', 'sum'),
            cantidad_promedio = ('cantidad',       np.mean),
        )
        .reset_index()
        .assign(cantidad_promedio=lambda d: d['cantidad_promedio'].round(2))
    )

    return input_data, output_data


if __name__ == '__main__':
    entrada, salida_esperada = generar_caso_de_uso_analizar_ventas()

    print("=== INPUT ===")
    print(f"Shape del DataFrame: {entrada['df'].shape}")
    print(entrada['df'].head())

    print("\n=== OUTPUT ESPERADO ===")
    print(f"Shape: {salida_esperada.shape}")
    print(salida_esperada)
