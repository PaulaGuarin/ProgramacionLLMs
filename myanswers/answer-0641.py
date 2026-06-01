import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


def aplicar_target_encoding(df, cat_cols, target_col):
    """
    Aplica Target Encoding out-of-fold (KFold k=5) a columnas categóricas.

    Por cada fold:
    - Calcula la media de target_col por categoría usando solo las filas
      del conjunto de entrenamiento (groupby + mean).
    - Asigna esa media a cada fila del conjunto de validación.
    - Si una categoría no aparece en entrenamiento, asigna la media global.

    Parámetros
    ----------
    df         : pd.DataFrame con columnas categóricas y columna target.
    cat_cols   : list[str] — columnas categóricas a codificar.
    target_col : str — columna objetivo binaria (0/1).

    Retorna
    -------
    pd.DataFrame con las columnas cat_cols reemplazadas por float (encodings).
    Las demás columnas e índice permanecen sin cambios.
    """
    n_rows = len(df)
    df_out = df.copy()
    kf = KFold(n_splits=5, shuffle=False)
    global_mean = df[target_col].mean()

    for col in cat_cols:
        encoded = np.zeros(n_rows, dtype=float)
        for train_idx, val_idx in kf.split(df):
            train_df = df.iloc[train_idx]
            means = train_df.groupby(col)[target_col].mean()
            for i in val_idx:
                cat = df.iloc[i][col]
                encoded[i] = means.get(cat, global_mean)
        df_out[col] = encoded

    return df_out


if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)

    n = 30
    df_demo = pd.DataFrame({
        'ciudad':    np.random.choice(['Bogota', 'Medellin', 'Cali'], n),
        'categoria': np.random.choice(['A', 'B', 'C'], n),
        'valor':     np.round(np.random.randn(n), 3),
        'target':    np.random.randint(0, 2, n),
    })

    resultado = aplicar_target_encoding(df_demo, ['ciudad', 'categoria'], 'target')
    print("Input (primeras 5 filas):")
    print(df_demo.head(5).to_string())
    print("\nOutput (primeras 5 filas):")
    print(resultado.head(5).to_string())
    print(f"\nTipo columnas codificadas:")
    print(resultado[['ciudad', 'categoria']].dtypes)
    print(f"\nRetorna DataFrame: {isinstance(resultado, pd.DataFrame)}")
