import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import random

def generar_caso_de_uso_importancia_permutacion():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función calcular_importancia_permutacion.
    """
    
    # 1. Configuración aleatoria de dimensiones
    n_rows = random.randint(40, 60)       # Entre 40 y 60 filas para que la permutación sea estable
    n_features = random.randint(3, 5)     # Entre 3 y 5 columnas de características
    
    # 2. Generar nombres de columnas aleatorios
    feature_cols = [f'metrica_{i}' for i in range(n_features)]
    target_col = 'es_fraude'
    
    # 3. Generar datos numéricos aleatorios
    # Creamos una matriz base con valores flotantes aleatorios
    data = np.random.randn(n_rows, n_features)
    df = pd.DataFrame(data, columns=feature_cols)
    
    # Generar la variable objetivo binaria
    # Para que el modelo aprenda algo real (y la importancia por permutación no sea cero o ruido),
    # hacemos que la variable objetivo dependa parcialmente de la primera columna.
    probabilidad_base = 1 / (1 + np.exp(-df[feature_cols[0]])) # Sigmoide simple
    df[target_col] = np.random.binomial(1, probabilidad_base)
    
    # ---------------------------------------------------------
    # 4. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        'df': df.copy(),
        'target_col': target_col
    }
    
    # ---------------------------------------------------------
    # 5. Calcular el OUTPUT esperado (Ground Truth)
    #    Replicamos exactamente los pasos solicitados en la misión
    # ---------------------------------------------------------
    
    # A. Separar X e y
    X_expected = df.drop(columns=[target_col])
    y_expected = df[target_col]
    
    # B. Entrenar el RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_expected, y_expected)
    
    # C. Calcular permutation_importance
    # Usamos random_state=42 para que el cálculo interno de la permutación sea determinista
    result = permutation_importance(rf, X_expected, y_expected, random_state=42)
    
    # D. Estructurar el diccionario de salida (claves: nombres, valores: importancias promedio)
    output_data = dict(zip(X_expected.columns, result.importances_mean))
    
    return input_data, output_data

# --- Ejemplo de uso y verificación ---
if __name__ == "__main__":
    # Generamos un caso de prueba
    entrada, salida_esperada = generar_caso_de_uso_importancia_permutacion()
    
    print("=== INPUT (Diccionario de Entrada) ===")
    print(f"Columna Objetivo: {entrada['target_col']}")
    print(f"Dimensiones del DataFrame: {entrada['df'].shape}")
    print("\nPrimeras 3 filas del DataFrame:")
    print(entrada['df'].head(3))
    
    print("\n=== OUTPUT ESPERADO (Ground Truth) ===")
    print("Diccionario de importancia por permutación (importances_mean):")
    for columna, importancia in salida_esperada.items():
        print(f"  - {columna}: {importancia:.4f}")
