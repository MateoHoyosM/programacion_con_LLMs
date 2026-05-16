# ============================================================
# SOLUCIÓN: preparar_datos
# ============================================================
import pandas as pd
import numpy as np
import random
from sklearn.impute import KNNImputer

def preparar_datos(df, target_col):
    """
    Prepara datos clínicos para modelado:

    Pasos:
      1. Convierte 'nivel_riesgo' (categórica) a enteros con .map().
      2. Separa X (características) e y (target).
      3. Imputa valores faltantes en X con KNNImputer(n_neighbors=5).
      4. Transforma y con np.log1p() para estabilizar su rango.

    Parámetros:
      df         : pd.DataFrame — datos clínicos crudos
      target_col : str          — nombre de la columna objetivo

    Retorna:
      Tupla (X_imputed: np.ndarray, y_transformed: np.ndarray)
    """

    df = df.copy()

    # ── Paso 1: Mapeo categórico ─────────────────────────────────────────────
    df['nivel_riesgo'] = df['nivel_riesgo'].map({'Bajo': 0, 'Medio': 1, 'Alto': 2})

    # ── Paso 2: Separar X e y ────────────────────────────────────────────────
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ── Paso 3: Imputación KNN ───────────────────────────────────────────────
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)

    # ── Paso 4: Transformación log1p sobre y ─────────────────────────────────
    y_transformed = np.log1p(y.to_numpy())

    return X_imputed, y_transformed


# ============================================================
# USE CASE GENERATOR (corregido artefactos de formato)
# ============================================================
def generar_caso_de_uso_preparar_datos_clinicos():
    n_rows = random.randint(20, 40)

    niveles = ['Bajo', 'Medio', 'Alto']
    df = pd.DataFrame({
        'edad':        np.random.randint(18, 80, n_rows).astype(float),
        'colesterol':  np.random.uniform(150, 300, n_rows),
        'nivel_riesgo': random.choices(niveles, k=n_rows)
    })

    # Introducir NaNs aleatorios en 'colesterol'
    mask = np.random.choice([True, False], size=n_rows, p=[0.2, 0.8])
    df.loc[mask, 'colesterol'] = np.nan

    target_col = 'costo_tratamiento'
    df[target_col] = np.random.lognormal(mean=8, sigma=2, size=n_rows)

    input_data = {'df': df.copy(), 'target_col': target_col}

    # ── Output esperado ──────────────────────────────────────────────────────
    df_expected = df.copy()
    df_expected['nivel_riesgo'] = df_expected['nivel_riesgo'].map({'Bajo': 0, 'Medio': 1, 'Alto': 2})

    X_expected = df_expected.drop(columns=[target_col])
    y_expected = df_expected[target_col]

    imputer   = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X_expected)
    y_trans   = np.log1p(y_expected.to_numpy())

    return input_data, (X_imputed, y_trans)


# ============================================================
# VALIDACIÓN AUTOMÁTICA
# ============================================================
def validar_solucion(n_casos=200, tolerancia=1e-9):
    errores = 0

    for i in range(n_casos):
        inp, (X_exp, y_exp) = generar_caso_de_uso_preparar_datos_clinicos()

        X_res, y_res = preparar_datos(inp['df'], inp['target_col'])

        # ── Verificar shapes ─────────────────────────────────────────────────
        if X_res.shape != X_exp.shape:
            print(f"[Caso {i+1}] ❌ Shape de X incorrecto: {X_res.shape} vs {X_exp.shape}")
            errores += 1
            continue

        if y_res.shape != y_exp.shape:
            print(f"[Caso {i+1}] ❌ Shape de y incorrecto: {y_res.shape} vs {y_exp.shape}")
            errores += 1
            continue

        # ── Verificar valores de X ───────────────────────────────────────────
        if not np.allclose(X_res, X_exp, atol=tolerancia):
            print(f"[Caso {i+1}] ❌ Valores de X imputada no coinciden")
            print(f"  Max diferencia: {np.abs(X_res - X_exp).max():.2e}")
            errores += 1
            continue

        # ── Verificar valores de y ───────────────────────────────────────────
        if not np.allclose(y_res, y_exp, atol=tolerancia):
            print(f"[Caso {i+1}] ❌ Valores de y transformada no coinciden")
            print(f"  Max diferencia: {np.abs(y_res - y_exp).max():.2e}")
            errores += 1

    # ── Resumen ──────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    if errores == 0:
        print(f"✅ Todos los {n_casos} casos pasaron correctamente.")
    else:
        print(f"❌ {errores} de {n_casos} casos fallaron.")
    print("="*50)


# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    inp, (X_exp, y_exp) = generar_caso_de_uso_preparar_datos_clinicos()

    print("── Input DataFrame ──")
    print(inp['df'].head(10))
    print(f"\nTarget col : '{inp['target_col']}'")
    print(f"NaNs en colesterol: {inp['df']['colesterol'].isna().sum()}")

    X_res, y_res = preparar_datos(inp['df'], inp['target_col'])

    print(f"\n── X imputada (shape={X_res.shape}) ──")
    print(X_res[:5])

    print(f"\n── y transformada (shape={y_res.shape}) ──")
    print(y_res[:5])

    print(f"\n── Coincide con esperado ──")
    print(f"  X: {np.allclose(X_res, X_exp)}")
    print(f"  y: {np.allclose(y_res, y_exp)}")

    print("\n── Validación con 200 casos aleatorios ──")
    validar_solucion(n_casos=200)
