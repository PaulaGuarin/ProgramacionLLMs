import pandas as pd
import numpy as np

def analizar_ventas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y agrega un DataFrame de ventas por mes y producto.

    Parámetros
    ----------
    df : pd.DataFrame
        Debe contener las columnas: fecha, producto, cantidad, precio_unitario.

    Retorna
    -------
    pd.DataFrame
        Agrupado por (mes, producto) con ingreso_total y cantidad_promedio.
    """
    # 1. Copia defensiva para no modificar el original
    df = df.copy()

    # 2. Eliminar filas con nulos en las columnas clave
    columnas_requeridas = ["fecha", "producto", "cantidad", "precio_unitario"]
    df.dropna(subset=columnas_requeridas, inplace=True)

    # 3. Convertir fecha a datetime y extraer el mes (periodo año-mes)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["mes"] = df["fecha"].dt.to_period("M")  # Ej: 2024-03

    # 4. Calcular ingreso total por fila con numpy
    df["ingreso_total"] = np.multiply(df["cantidad"], df["precio_unitario"])

    # 5. Agrupar por mes y producto
    resumen = (
        df.groupby(["mes", "producto"], sort=True)
        .agg(
            ingreso_total=("ingreso_total", "sum"),
            cantidad_promedio=("cantidad", np.mean),
        )
        .reset_index()
    )

    # 6. Redondear cantidad_promedio a 2 decimales
    resumen["cantidad_promedio"] = resumen["cantidad_promedio"].round(2)

    return resumen

    # Dataset de prueba
data = {
    "fecha": ["2024-01-05", "2024-01-20", "2024-02-10", "2024-02-15", None,  "2024-01-08"],
    "producto": ["Laptop", "Mouse",  "Laptop", "Teclado", "Mouse",  "Mouse"],
    "cantidad": [2,        10,        3,        5,         8,         None],
    "precio_unitario": [1500, 25,    1500,     45,        25,        25],
}

df = pd.DataFrame(data)
resultado = analizar_ventas(df)
print(resultado)