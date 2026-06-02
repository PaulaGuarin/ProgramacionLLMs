import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

def calcular_importancia_permutacion(df, target_col):
    """
    Esta es la lógica que el evaluador ejecutará internamente
    o comparará contra la solución del estudiante.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Quitamos n_repeats=5 para usar el comportamiento por defecto de sklearn,
    # a menos que tu rúbrica exija explícitamente 5 repeticiones.
    result = permutation_importance(model, X, y, random_state=42)

    return {col: imp for col, imp in zip(X.columns, result.importances_mean)}


def generar_caso_de_uso_calcular_importancia_permutacion(df=None, target_col=None, **kwargs):
    """
    Generador adaptado al sistema de evaluación automatizado.
    """
    # -------------------------------------------------------------------------
    # CASO A: El evaluador pasa argumentos -> Actúa como la solución esperada (devuelve DICT)
    # -------------------------------------------------------------------------
    if df is not None:
        # Aseguramos capturar target_col correctamente si viene en kwargs
        t_col = target_col if target_col is not None else kwargs.get('target_col')
        return calcular_importancia_permutacion(df, t_col)

    # -------------------------------------------------------------------------
    # CASO B: Sin argumentos -> Actúa como GENERADOR de datos de prueba (devuelve TUPLE)
    # -------------------------------------------------------------------------
    n = random.randint(30, 80)
    m = random.randint(3, 6)

    X = np.random.randn(n, m)
    cols = [f"f{i}" for i in range(m)]
    y = (X[:, 0] + np.random.randn(n) > 0).astype(int)

    df_gen = pd.DataFrame(X, columns=cols)
    target_col_gen = "target"
    df_gen[target_col_gen] = y

    # Construimos el diccionario INPUT requerido por la plataforma
    input_data = {
        "df": df_gen.copy(), 
        "target_col": target_col_gen
    }

    # Calculamos el OUTPUT esperado (Ground Truth) usando la función de arriba
    output_data = calcular_importancia_permutacion(df_gen, target_col_gen)

    return input_data, output_data
