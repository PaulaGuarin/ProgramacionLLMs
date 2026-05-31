import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import random


def generar_caso_de_uso_aplicar_target_encoding():
    """
    Genera un caso de uso aleatorio para aplicar_target_encoding.

    Variaciones cubiertas:
    - n_rows       : 25 – 60 filas.
    - cat_cols     : 1 – 3 columnas categóricas de un pool variado.
    - cardinalidad : 2 – 4 valores únicos por columna.
    - cols extra   : 0 – 2 columnas numéricas que no deben modificarse.
    - target_col   : siempre 'target', proporción de positivos 20 %–80 %.
    - Caso edge    : 30 % de prob. de inyectar una categoría '__RARO__' que
                     solo aparece en el último fold → fuerza fallback a media global.

    Retorna
    -------
    input_dict : dict con claves 'df', 'cat_cols', 'target_col'
    df_out     : pd.DataFrame con las columnas cat_cols codificadas (ground truth)
    """

    n_rows = random.randint(25, 60)
    positive_rate = random.uniform(0.2, 0.8)

    pool_categoricas = {
        'ciudad':    ['Bogota', 'Medellin', 'Cali', 'Barranquilla', 'Cartagena',
                      'Bucaramanga', 'Pereira', 'Manizales'],
        'categoria': ['A', 'B', 'C', 'D', 'E'],
        'region':    ['Norte', 'Sur', 'Oriente', 'Occidente', 'Centro'],
        'canal':     ['Online', 'Tienda', 'Telefono', 'App'],
        'segmento':  ['Premium', 'Estandar', 'Basico', 'Trial'],
    }

    n_cat_cols = random.randint(1, 3)
    cat_col_names = random.sample(list(pool_categoricas.keys()), n_cat_cols)

    data = {}
    for col_name in cat_col_names:
        valores = pool_categoricas[col_name]
        k = random.randint(2, min(4, len(valores)))
        data[col_name] = np.random.choice(random.sample(valores, k), n_rows)

    n_num_extra = random.randint(0, 2)
    for col_name in random.sample(['ingreso', 'edad', 'score', 'cantidad', 'dias'],
                                  n_num_extra):
        data[col_name] = np.round(np.random.randn(n_rows), 3)

    data['target'] = (np.random.rand(n_rows) < positive_rate).astype(int)
    df = pd.DataFrame(data)

    # Caso edge: categoría rara en el último fold
    if random.random() < 0.3:
        kf_tmp = KFold(n_splits=5, shuffle=False)
        folds = list(kf_tmp.split(df))
        _, last_val_idx = folds[-1]
        rare_col = cat_col_names[0]
        df.iloc[last_val_idx[0], df.columns.get_loc(rare_col)] = '__RARO__'

    # Ground truth
    target_col = 'target'
    df_out = df.copy()
    kf = KFold(n_splits=5, shuffle=False)
    global_mean = df[target_col].mean()

    for col in cat_col_names:
        encoded = np.zeros(n_rows, dtype=float)
        for train_idx, val_idx in kf.split(df):
            train_df = df.iloc[train_idx]
            means = train_df.groupby(col)[target_col].mean()
            for i in val_idx:
                cat = df.iloc[i][col]
                encoded[i] = means.get(cat, global_mean)
        df_out[col] = encoded

    input_dict = {
        'df':         df.copy(),
        'cat_cols':   cat_col_names,
        'target_col': target_col,
    }
    return input_dict, df_out


if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_aplicar_target_encoding()
    print("=== INPUT ===")
    print(f"cat_cols: {entrada['cat_cols']},  target_col: '{entrada['target_col']}'")
    print(entrada['df'].head(8).to_string())
    print(f"\nShape: {entrada['df'].shape}")
    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada.head(8).to_string())
    print(f"\nDtypes de columnas codificadas:")
    print(salida_esperada[entrada['cat_cols']].dtypes)
