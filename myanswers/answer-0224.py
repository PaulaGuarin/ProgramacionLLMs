import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random

def generar_caso_de_uso_deteccion_fraude():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función preparar_deteccion_fraude.
    """
    
    # 1. Configuración aleatoria de dimensiones
    n_rows = random.randint(10, 20)  # Entre 10 y 20 transacciones
    
    # 2. Generar datos aleatorios de transacciones
    # Montos aleatorios entre 10 y 2000
    montos = np.random.uniform(10.0, 2000.0, size=n_rows)
    
    # Categorías fijas para tipo de comercio
    categorias_comercio = ["Retail", "Food", "Tech", "Travel", "Entertainment"]
    tipos_comercio = np.random.choice(categorias_comercio, size=n_rows)
    
    # Crear el DataFrame inicial
    df_transacciones = pd.DataFrame({
        'monto': montos,
        'tipo_comercio': tipos_comercio
    })
    
    # Introducir algunos NaNs aleatorios en la columna 'monto' (aprox 15% de probabilidad)
    mask_nan = np.random.choice([True, False], size=n_rows, p=[0.15, 0.85])
    df_transacciones.loc[mask_nan, 'monto'] = np.nan
    
    # Definir un monto máximo normal aleatorio para el caso de prueba
    monto_max_normal = float(random.randint(800, 1500))
    
    # ---------------------------------------------------------
    # 3. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        'df_transacciones': df_transacciones.copy(),
        'monto_max_normal': monto_max_normal
    }
    
    # ---------------------------------------------------------
    # 4. Calcular el OUTPUT esperado (Ground Truth)
    # ---------------------------------------------------------
    df_expected = df_transacciones.copy()
    
    # A. Limpieza de montos (Imputar con la Mediana)
    # Nota: Usamos reshape(-1, 1) porque SimpleImputer espera una matriz 2D
    imputer = SimpleImputer(strategy='median')
    df_expected['monto'] = imputer.fit_transform(df_expected[['monto']])
    
    # B. Etiquetado Automático (Antes de escalar el monto)
    # 1 si monto > monto_max_normal else 0
    y_expected = np.where(df_expected['monto'] > monto_max_normal, 1, 0)
    
    # C. Codificación de 'tipo_comercio' con LabelEncoder
    le = LabelEncoder()
    df_expected['tipo_comercio'] = le.fit_transform(df_expected['tipo_comercio'])
    
    # D. Normalización de 'monto' con StandardScaler
    scaler = StandardScaler()
    df_expected['monto'] = scaler.fit_transform(df_expected[['monto']])
    
    # E. Dar formato a la salida (Array con 'tipo_comercio' y 'monto', y Array de 'sospechoso')
    # Reordenamos
