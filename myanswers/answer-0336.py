import pandas as pd
import numpy as np
import random


# ── Implementación de referencia (ground truth) ───────────────────────────────

def _reference_calcular_estadisticas(df):
    medias = df.mean().to_numpy()
    desviaciones = df.std().to_numpy()
    return medias, desviaciones


# ── Generador principal ───────────────────────────────────────────────────────

def generar_caso_de_uso_calcular_estadisticas():
    """
    Genera un caso de uso aleatorio para calcular_estadisticas.

    Variaciones cubiertas:
    - n_rows      : 10 – 200 filas.
    - n_cols      : 2 – 8 columnas numéricas.
    - nombres     : subconjunto aleatorio del pool para evitar nombres fijos.
    - distribución: cada columna tiene media y escala distintas (no centradas
                    en 0) para que las medias y stds varíen entre columnas.
    - dtypes       : mezcla aleatoria de float64 e int64 (ambos son numéricos).
    - Caso edge columna constante: con prob. 0.2 una columna tiene todos los
                    valores iguales → std = 0 (prueba que no hay división por 0).

    Retorna
    -------
    input_dict : dict con clave 'df'
    (medias, desviaciones) : tupla de numpy arrays (ground truth)
    """

    # ── 1. Parámetros estructurales ───────────────────────────────────────────
    n_rows = random.randint(10, 200)
    n_cols = random.randint(2, 8)

    pool_nombres = [
        'edad', 'ingreso', 'score', 'deuda', 'saldo', 'monto',
        'frecuencia', 'dias', 'cantidad', 'ratio', 'altura', 'peso',
    ]
    col_names = random.sample(pool_nombres, n_cols)

    # ── 2. Generar datos con medias y escalas variadas ────────────────────────
    data = {}
    for col in col_names:
        media_real = random.uniform(-100, 500)
        escala     = random.uniform(0.5, 50)
        usar_int   = random.random() < 0.3     # 30% de columnas como int

        valores = np.random.randn(n_rows) * escala + media_real
        if usar_int:
            valores = valores.astype(int).astype(float)  # int → float para consistencia
        data[col] = np.round(valores, 4)

    df = pd.DataFrame(data)

    # ── 3. Caso edge: columna constante (std = 0) ─────────────────────────────
    if random.random() < 0.2:
        col_constante = random.choice(col_names)
        valor_fijo = round(random.uniform(-50, 200), 2)
        df[col_constante] = valor_fijo

    # ── 4. Calcular ground truth ──────────────────────────────────────────────
    input_dict = {'df': df.copy()}
    medias, desviaciones = _reference_calcular_estadisticas(df.copy())

    return input_dict, (medias, desviaciones)


# ── Suite de validación ───────────────────────────────────────────────────────

def validar_solucion(func, n_trials=20, seed=None):
    """
    Ejecuta n_trials casos aleatorios contra la función del estudiante.

    Verificaciones por trial:
    1. Retorna una tupla de longitud 2.
    2. Ambos elementos son numpy.ndarray.
    3. Ambos arrays tienen shape (n_cols,).
    4. Los valores de medias coinciden con el ground truth (atol=1e-9).
    5. Los valores de desviaciones coinciden con el ground truth (atol=1e-9).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    resultados = []
    for i in range(n_trials):
        entrada, (medias_esp, stds_esp) = generar_caso_de_uso_calcular_estadisticas()
        df_in  = entrada['df']
        n_cols = df_in.shape[1]
        tiene_constante = any(df_in[c].nunique() == 1 for c in df_in.columns)

        try:
            resultado = func(df_in.copy())
            errores = []

            # 1. Tupla de 2 elementos
            if not (isinstance(resultado, tuple) and len(resultado) == 2):
                errores.append(f"debe retornar tupla de 2, recibió {type(resultado)}")
            else:
                medias_res, stds_res = resultado

                # 2. Tipos numpy
                if not isinstance(medias_res, np.ndarray):
                    errores.append(f"medias debe ser ndarray, es {type(medias_res)}")
                if not isinstance(stds_res, np.ndarray):
                    errores.append(f"desviaciones debe ser ndarray, es {type(stds_res)}")

                if isinstance(medias_res, np.ndarray) and isinstance(stds_res, np.ndarray):

                    # 3. Shapes
                    if medias_res.shape != (n_cols,):
                        errores.append(f"medias.shape={medias_res.shape}, esperado ({n_cols},)")
                    if stds_res.shape != (n_cols,):
                        errores.append(f"desviaciones.shape={stds_res.shape}, esperado ({n_cols},)")

                    # 4 & 5. Valores correctos
                    if not errores:
                        if not np.allclose(medias_res, medias_esp, atol=1e-9):
                            diff = np.abs(medias_res - medias_esp).max()
                            errores.append(f"medias incorrectas (diff max={diff:.2e})")
                        if not np.allclose(stds_res, stds_esp, atol=1e-9):
                            diff = np.abs(stds_res - stds_esp).max()
                            errores.append(f"desviaciones incorrectas (diff max={diff:.2e})")

            passed = len(errores) == 0
            nota   = "; ".join(errores) if errores else "OK"

        except Exception as exc:
            passed = False
            nota   = f"EXCEPCIÓN: {exc}"

        resultados.append({
            'trial':     i + 1,
            'n_rows':    df_in.shape[0],
            'n_cols':    n_cols,
            'constante': tiene_constante,
            'passed':    passed,
            'nota':      nota,
        })

    # ── Reporte ───────────────────────────────────────────────────────────────
    total  = len(resultados)
    passed = sum(r['passed'] for r in resultados)
    print(f"\n{'='*60}")
    print(f"  RESULTADOS: {passed}/{total} trials correctos")
    print(f"{'='*60}")
    for r in resultados:
        icono     = "✓" if r['passed'] else "✗"
        constante = " [col_cte]" if r['constante'] else ""
        print(
            f"  {icono} Trial {r['trial']:02d} | {r['n_rows']:3d} filas | "
            f"{r['n_cols']} cols{constante} | {r['nota']}"
        )
    print(f"{'='*60}\n")
    return passed == total


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Validando implementación de referencia ──")
    validar_solucion(_reference_calcular_estadisticas, n_trials=20, seed=7)

    # Demo visual
    random.seed(1); np.random.seed(1)
    entrada, (medias, stds) = generar_caso_de_uso_calcular_estadisticas()
    df_in = entrada['df']

    print("=== INPUT ===")
    print(f"Shape   : {df_in.shape}")
    print(f"Columnas: {list(df_in.columns)}")
    print(df_in.head(6).to_string())

    print("\n=== OUTPUT ESPERADO ===")
    print(f"medias       (shape {medias.shape}): {np.round(medias, 4)}")
    print(f"desviaciones (shape {stds.shape}): {np.round(stds, 4)}")
    print(f"\nDetalle por columna:")
    for col, m, s in zip(df_in.columns, medias, stds):
        print(f"  {col:<15} media={m:+10.4f}   std={s:.4f}")
