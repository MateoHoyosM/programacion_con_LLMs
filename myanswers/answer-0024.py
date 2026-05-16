# ============================================================
# SOLUCIÓN: normalizar_expresion
# ============================================================
import pandas as pd
import numpy as np

def normalizar_expresion(df, threshold):
    """
    Normaliza datos de expresión génica (RNA-Seq raw counts).

    Pasos:
      1. Filtra genes cuyo promedio de conteos sea estrictamente mayor que threshold.
      2. Aplica transformación log2(x + 1) para estabilizar la varianza.
      3. Calcula el Z-score por gen (columna), usando ddof=1.

    Parámetros:
      df        : pd.DataFrame — filas = muestras, columnas = genes (conteos crudos)
      threshold : float        — umbral mínimo de expresión promedio

    Retorna:
      pd.DataFrame con genes filtrados, transformados y normalizados.
      DataFrame vacío si ningún gen supera el threshold.
    """

    # ── Paso 1: Filtrar genes por expresión mínima ──────────────────────────
    means = df.mean(axis=0)
    genes_to_keep = means[means > threshold].index
    df_filtered = df[genes_to_keep].copy()

    if df_filtered.empty:
        return pd.DataFrame()

    # ── Paso 2: Transformación logarítmica log2(x + 1) ─────────────────────
    df_log = np.log2(df_filtered + 1)

    # ── Paso 3: Z-score por gen (columna), ddof=1 ──────────────────────────
    mean_log = df_log.mean(axis=0)
    std_log  = df_log.std(axis=0, ddof=1)

    df_zscore = (df_log - mean_log) / std_log
    # Genes con std=0 (todos los valores iguales) quedan como NaN — comportamiento esperado

    return df_zscore


# ============================================================
# USE CASE GENERATOR (copiado tal cual del enunciado)
# ============================================================
def generar_caso_de_uso_normalizar_expresion():
    n_samples = np.random.randint(5, 15)
    n_genes   = np.random.randint(5, 15)

    data         = np.random.randint(0, 100, size=(n_samples, n_genes))
    gene_names   = [f"Gene_{i}" for i in range(n_genes)]
    sample_names = [f"Sample_{i}" for i in range(n_samples)]

    df        = pd.DataFrame(data, index=sample_names, columns=gene_names)
    threshold = np.random.uniform(5, 30)

    input_data = {"df": df, "threshold": threshold}

    # Output esperado
    means        = df.mean()
    genes_to_keep = means[means > threshold].index
    df_filtered  = df[genes_to_keep].copy()

    if df_filtered.empty:
        expected_output = pd.DataFrame()
    else:
        df_log    = np.log2(df_filtered + 1)
        mean_log  = df_log.mean(axis=0)
        std_log   = df_log.std(axis=0, ddof=1)
        expected_output = (df_log - mean_log) / std_log

    return input_data, expected_output


# ============================================================
# VALIDACIÓN AUTOMÁTICA — corre N casos aleatorios
# ============================================================
def validar_solucion(n_casos=200, tolerancia=1e-10):
    """Compara la salida de normalizar_expresion con el output esperado del generador."""
    errores = 0

    for i in range(n_casos):
        inp, expected = generar_caso_de_uso_normalizar_expresion()
        result = normalizar_expresion(inp["df"], inp["threshold"])

        # ── Caso DataFrame vacío ────────────────────────────────────────────
        if expected.empty:
            if not result.empty:
                print(f"[Caso {i+1}] ❌ Se esperaba DataFrame vacío, se obtuvo:\n{result}")
                errores += 1
            continue

        # ── Verificar shape ─────────────────────────────────────────────────
        if result.shape != expected.shape:
            print(f"[Caso {i+1}] ❌ Shape incorrecto: obtenido {result.shape}, esperado {expected.shape}")
            errores += 1
            continue

        # ── Verificar columnas ──────────────────────────────────────────────
        if not result.columns.equals(expected.columns):
            print(f"[Caso {i+1}] ❌ Columnas incorrectas: {result.columns.tolist()} vs {expected.columns.tolist()}")
            errores += 1
            continue

        # ── Verificar valores numéricos (tolerando NaN en std=0) ────────────
        nan_match = result.isna().equals(expected.isna())
        if not nan_match:
            print(f"[Caso {i+1}] ❌ Patrón de NaN no coincide")
            errores += 1
            continue

        # Comparar solo donde no hay NaN
        mask = ~expected.isna()
        if not np.allclose(result[mask], expected[mask], atol=tolerancia):
            print(f"[Caso {i+1}] ❌ Valores numéricos incorrectos")
            print(f"  Obtenido:\n{result}")
            print(f"  Esperado:\n{expected}")
            errores += 1

    # ── Resumen ─────────────────────────────────────────────────────────────
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
    # Mostrar un ejemplo concreto
    np.random.seed(42)
    inp, expected = generar_caso_de_uso_normalizar_expresion()

    print("── Input DataFrame ──")
    print(inp["df"])
    print(f"\nThreshold: {inp['threshold']:.4f}")

    result = normalizar_expresion(inp["df"], inp["threshold"])

    print("\n── Resultado obtenido ──")
    print(result)
    print("\n── Resultado esperado ──")
    print(expected)

    # Validación masiva
    print("\n── Validación con 200 casos aleatorios ──")
    validar_solucion(n_casos=200)
