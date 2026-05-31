import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import random


def generar_caso_de_uso_calcular_importancia_permutacion(**kwargs):
    n_rows     = random.randint(80, 300)
    n_features = random.randint(3, 8)

    pool_nombres = ['edad', 'ingreso', 'score_credito', 'dias_cliente',
                    'num_transacciones', 'monto_promedio', 'deuda_actual',
                    'num_productos', 'frecuencia_login', 'dias_ultimo_pago',
                    'saldo_promedio', 'ratio_uso']
    feature_names = random.sample(pool_nombres, n_features)
    target_col    = random.choice(['target', 'label', 'fraude', 'churn'])

    n_signal    = random.randint(1, min(3, n_features))
    signal_cols = random.sample(feature_names, n_signal)
    latent      = np.random.randn(n_rows)
    data        = {}
    for col in feature_names:
        noise = np.random.randn(n_rows)
        if col in signal_cols:
            data[col] = np.round(latent * random.uniform(0.6, 1.5) + noise * 0.3, 4)
        else:
            data[col] = np.round(noise, 4)

    pos_rate    = random.uniform(0.10, 0.20) if random.random() < 0.3 else random.uniform(0.35, 0.65)
    signal_sum  = sum(data[c] for c in signal_cols)
    prob        = 1 / (1 + np.exp(-signal_sum))
    target_vals = (prob > (1 - pos_rate)).astype(int)

    if target_vals.sum() < 5:
        target_vals[np.random.choice(np.where(target_vals == 0)[0], 5, replace=False)] = 1
    if (1 - target_vals).sum() < 5:
        target_vals[np.random.choice(np.where(target_vals == 1)[0], 5, replace=False)] = 0

    data[target_col] = target_vals
    df = pd.DataFrame(data)

    # Ground truth
    X      = df.drop(columns=[target_col])
    y      = df[target_col]
    model  = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    result = permutation_importance(model, X, y, random_state=42)
    output_data = {col: imp for col, imp in zip(X.columns, result.importances_mean)}

    input_dict = {'df': df.copy(), 'target_col': target_col}
    return input_dict, output_data


if __name__ == "__main__":
    entrada, esperado = generar_caso_de_uso_calcular_importancia_permutacion()
    df_in      = entrada['df']
    target_col = entrada['target_col']
    feat_cols  = [c for c in df_in.columns if c != target_col]
    print("=== INPUT ===")
    print(f"target_col : '{target_col}'")
    print(f"Shape      : {df_in.shape}")
    print(f"Features   : {feat_cols}")
    print(f"Positivos  : {df_in[target_col].mean():.1%}")
    print(df_in.head(6).to_string())
    print("\n=== OUTPUT ESPERADO (dict) ===")
    for k, v in sorted(esperado.items(), key=lambda x: -x[1]):
        bar = "█" * int(max(0, v) * 200)
        print(f"  {k:<25} {v:+.6f}  {bar}")
