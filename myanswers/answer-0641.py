import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import random

def generar_caso_de_uso_target_encoding():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función aplicar_target_encoding.
    """
    
    # 1. Configuración aleatoria de dimensiones
    n_rows = random.randint(25, 35)  # Entre 25 y 35 filas para que los 5 folds sean consistentes
    
    # 2. Generar datos aleatorios categóricos y numéricos
    # Categorías para simular las columnas
    cat_pool_1 = ['Alta', 'Media', 'Baja', 'Critica']
    cat_pool_2 = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']
    
    col_cat_1 = np.random.choice(cat_pool_1, size=n_rows)
    col_cat_2 = np.random.choice(cat_pool_2, size=n_rows)
    col_num = np.random.uniform(10, 100, size=n_rows) # Columna extra que no debe modificarse
    
    # Variable objetivo binaria (0 o 1)
    target = np.random.choice([0, 1], size=n_rows, p=[0.4, 0.6])
    
    # Crear el DataFrame inicial
    df = pd.DataFrame({
        'prioridad': col_cat_1,
        'region': col_cat_2,
        'monto': col_num,
        'objetivo': target
    })
    
    cat_cols = ['prioridad', 'region']
    target_col = 'objetivo'
    
    # Forzar un caso extremo intencional: añadir una categoría única al final del DataFrame
    # que probablemente no aparezca en el set de entrenamiento de algún fold para probar la media global.
    df.loc[n_rows - 1, 'prioridad'] = 'Nueva_Categoria_Rara'
    
    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(),
        'cat_cols': cat_cols,
        'target_col': target_col
    }
    
    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    # ---------------------------------------------------------
    df_expected = df.copy()
    global_mean = df_expected[target_col].mean()
    
    # Inicializar KFold tal como lo pide la consigna
    kf = KFold(n_splits=5, shuffle=False)
    
    for col in cat_cols:
        # Creamos una serie temporal para almacenar las codificaciones de esta columna
        encoded_series = pd.Series(index=df_expected.index, dtype=float)
        
        for train_idx, val_idx in kf.split(df_expected):
            # Dividir en entrenamiento y validación del fold
            df_train = df_expected.iloc[train_idx]
            df_val = df_expected.iloc[val_idx]
            
            # Calcular la media del target por categoría en el set de entrenamiento
            means_by_cat = df_train.groupby(col)[target_col].mean()
            
            # Mapear las medias a las categorías del set de validación
            fold_encoded = df_val[col].map(means_by_cat)
            
            # Si hay NaN (categorías de validación que no estaban en entrenamiento), rellenar con la media global
            fold_encoded = fold_encoded.fillna(global_mean)
            
            # Guardar el resultado en la serie temporal
            encoded_series.iloc[val_idx] = fold_encoded
            
        # Reemplazar la columna original por la serie codificada como float
        df_expected[col] = encoded_series
        
    output_data = df_expected
    
    return input_data, output_data

# --- Ejemplo de uso y verificación ---
if __name__ == "__main__":
    # Generamos un caso de prueba
    entrada, salida_esperada = generar_caso_de_uso_target_encoding()
    
    print("=== INPUT (Datos de entrada) ===")
    print(f"Columnas Categóricas a codificar: {entrada['cat_cols']}")
    print(f"Columna Objetivo: {entrada['target_col']}\n")
    print("DataFrame Original (Primeras 8 filas):")
    print(entrada['df'].head(8))
    
    print("\n=== OUTPUT ESPERADO (DataFrame codificado) ===")
    print("Nota cómo 'prioridad' y 'region' ahora son floats y 'monto' no cambió:")
    print(salida_esperada.head(8))
    
    # Verificación rápida del caso raro al final del DataFrame
    print("\nÚltima fila (verificación de categoría rara -> debe tener la media global):")
    print(salida_esperada.tail(1))
