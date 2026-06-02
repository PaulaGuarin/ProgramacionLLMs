import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import random

def aplicar_target_encoding_referencia(df, cat_cols, target_col):
    """
    Implementación de referencia exacta que resuelve el problema.
    """
    df_expected = df.copy()
    global_mean = df_expected[target_col].mean()
    
    kf = KFold(n_splits=5, shuffle=False)
    
    for col in cat_cols:
        encoded_series = pd.Series(index=df_expected.index, dtype=float)
        
        for train_idx, val_idx in kf.split(df_expected):
            df_train = df_expected.iloc[train_idx]
            df_val = df_expected.iloc[val_idx]
            
            # Calcular la media del target por categoría en el entrenamiento del fold
            means_by_cat = df_train.groupby(col)[target_col].mean()
            
            # Mapear las medias a las categorías del conjunto de validación
            fold_encoded = df_val[col].map(means_by_cat)
            
            # Rellenar con la media global si la categoría no existía en el entrenamiento
            fold_encoded = fold_encoded.fillna(global_mean)
            
            encoded_series.iloc[val_idx] = fold_encoded
            
        df_expected[col] = encoded_series
        
    return df_expected


def generar_caso_de_uso_aplicar_target_encoding(df=None, cat_cols=None, target_col=None, **kwargs):
    """
    Generador y validador oficial para la plataforma.
    """
    # -------------------------------------------------------------------------
    # CASO A: Se llama CON argumentos (Evaluación de la solución del estudiante)
    # -------------------------------------------------------------------------
    if df is not None:
        # Extraer parámetros por si vienen de manera posicional o por kwargs
        c_cols = cat_cols if cat_cols is not None else kwargs.get('cat_cols')
        t_col = target_col if target_col is not None else kwargs.get('target_col')
        return aplicar_target_encoding_referencia(df, c_cols, t_col)

    # -------------------------------------------------------------------------
    # CASO B: Se llama SIN argumentos (Generación de un caso aleatorio nuevo)
    # -------------------------------------------------------------------------
    n_rows = random.randint(25, 35)
    
    cat_pool_1 = ['Alta', 'Media', 'Baja', 'Critica']
    cat_pool_2 = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']
    
    col_cat_1 = np.random.choice(cat_pool_1, size=n_rows)
    col_cat_2 = np.random.choice(cat_pool_2, size=n_rows)
    col_num = np.random.uniform(10, 100, size=n_rows)
    
    target = np.random.choice([0, 1], size=n_rows, p=[0.4, 0.6])
    
    df_gen = pd.DataFrame({
        'prioridad': col_cat_1,
        'region': col_cat_2,
        'monto': col_num,
        'objetivo': target
    })
    
    cat_cols_gen = ['prioridad', 'region']
    target_col_gen = 'objetivo'
    
    # Forzar el caso extremo (Edge case de categoría huérfana en folds)
    df_gen.loc[n_rows - 1, 'prioridad'] = 'Nueva_Categoria_Rara'
    
    # Construir el diccionario INPUT
    input_data = {
        'df': df_gen.copy(),
        'cat_cols': cat_cols_gen,
        'target_col': target_col_gen
    }
    
    # Construir el OUTPUT esperado invocando la lógica de referencia
    output_data = aplicar_target_encoding_referencia(df_gen, cat_cols_gen, target_col_gen)
    
    return input_data, output_data

# --- Bloque de ejecución local para pruebas ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_aplicar_target_encoding()
    
    print("=== INPUT (Datos de entrada) ===")
    print(f"cat_cols: {entrada['cat_cols']}, target_col: '{entrada['target_col']}'")
    print(entrada['df'].head(5))
    
    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada.head(5))
