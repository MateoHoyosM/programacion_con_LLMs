# ============================================================
# SOLUCIÓN: calcular_pr_auc_cv
# ============================================================
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def calcular_pr_auc_cv(df, target_col, cv_folds=5):
    """
    Calcula el PR-AUC promedio mediante validación cruzada.

    Pasos:
      1. Separa X e y desde df.
      2. Escala X con StandardScaler.
      3. Instancia LogisticRegression(class_weight='balanced',
                                      max_iter=1000, random_state=42).
      4. Evalúa con cross_val_score(scoring='average_precision', cv=cv_folds).
      5. Retorna el promedio de los scores como float.

    Parámetros:
      df         : pd.DataFrame — dataset completo
      target_col : str          — nombre de la columna objetivo
      cv_folds   : int          — número de folds (default=5)

    Retorna:
      float — PR-AUC promedio sobre los folds
    """

    # ── Paso 1: Separar X e y ────────────────────────────────────────────────
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # ── Paso 2: Escalar características ──────────────────────────────────────
    X_scaled = StandardScaler().fit_transform(X)

    # ── Paso 3: Instanciar modelo ─────────────────────────────────────────────
    modelo = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
    )

    # ── Paso 4: Validación cruzada con PR-AUC ────────────────────────────────
    scores = cross_val_score(
        modelo, X_scaled, y,
        cv=cv_folds,
        scoring='average_precision',
    )

    # ── Paso 5-6: Promedio y retorno como float ───────────────────────────────
    return float(scores.mean())


# ============================================================
# USE CASE GENERATOR (copiado tal cual del enunciado)
# ============================================================
def generar_caso_de_uso_calcular_pr_auc_cv():
    rng = np.random.default_rng()

    n_samples     = int(rng.integers(200, 701))
    n_features    = int(rng.integers(3, 11))
    n_informative = int(rng.integers(2, n_features + 1))
    n_redundant   = int(rng.integers(0, max(1, (n_features - n_informative) + 1)))

    minority_frac = float(rng.uniform(0.05, 0.35))
    weights       = [round(1 - minority_frac, 2), round(minority_frac, 2)]
    dataset_seed  = int(rng.integers(0, 10_000))

    X_raw, y_raw = make_classification(
        n_samples     = n_samples,
        n_features    = n_features,
        n_informative = n_informative,
        n_redundant   = n_redundant,
        weights       = weights,
        flip_y        = 0.01,
        random_state  = dataset_seed,
    )

    cv_folds   = int(rng.choice([3, 5, 7, 10]))
    target_col = "target"
    col_names  = [f"feature_{i}" for i in range(n_features)]

    df = pd.DataFrame(X_raw, columns=col_names)
    df[target_col] = y_raw

    caso_input = {"df": df, "target_col": target_col, "cv_folds": cv_folds}

    # Output esperado
    X        = df.drop(columns=[target_col]).values
    y        = df[target_col].values
    X_scaled = StandardScaler().fit_transform(X)
    modelo   = LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42
    )
    scores       = cross_val_score(modelo, X_scaled, y,
                                   cv=cv_folds, scoring='average_precision')
    caso_output  = float(scores.mean())

    return caso_input, caso_output


# ============================================================
# VALIDACIÓN AUTOMÁTICA
# ============================================================
def validar_solucion(n_casos=200, tolerancia=1e-9):
    errores = 0

    for i in range(n_casos):
        inp, expected = generar_caso_de_uso_calcular_pr_auc_cv()

        try:
            result = calcular_pr_auc_cv(
                inp["df"], inp["target_col"], inp["cv_folds"]
            )
        except Exception as e:
            print(f"[Caso {i+1}] ❌ Excepción: {e}")
            errores += 1
            continue

        # ── Verificar tipo ───────────────────────────────────────────────────
        if not isinstance(result, float):
            print(f"[Caso {i+1}] ❌ Tipo incorrecto: {type(result).__name__}, se esperaba float")
            errores += 1
            continue

        # ── Verificar valor ──────────────────────────────────────────────────
        if not np.isclose(result, expected, atol=tolerancia):
            print(f"[Caso {i+1}] ❌ Valor incorrecto: obtenido={result:.8f}, "
                  f"esperado={expected:.8f}, diff={abs(result - expected):.2e}")
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
    # Ejemplo concreto
    inp, expected = generar_caso_de_uso_calcular_pr_auc_cv()

    clase_counts  = inp['df'][inp['target_col']].value_counts().to_dict()
    minority_pct  = round(100 * min(clase_counts.values()) / sum(clase_counts.values()), 1)

    print("── Input ──")
    print(f"  df.shape   : {inp['df'].shape}")
    print(f"  desbalance : clase minoritaria = {minority_pct}%  {clase_counts}")
    print(f"  cv_folds   : {inp['cv_folds']}")

    result = calcular_pr_auc_cv(inp["df"], inp["target_col"], inp["cv_folds"])

    print(f"\n── Resultado obtenido : {result:.8f}")
    print(f"── Resultado esperado : {expected:.8f}")
    print(f"── Coincide           : {np.isclose(result, expected)}")

    print("\n── Validación con 200 casos aleatorios ──")
    validar_solucion(n_casos=200)
