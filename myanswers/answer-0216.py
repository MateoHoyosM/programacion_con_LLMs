# ============================================================
# SOLUCIÓN CORREGIDA: generar_segmentos
# ============================================================
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans


def generar_segmentos(df, test_size, n_clusters):
    """
    Pipeline de segmentación de clientes.

    Pasos:
      1. Elimina filas con 'cluster' nulo.
      2. Imputa 'income' con la media por grupo de age (rangos de 10 años).
      3. Escala características con RobustScaler.
      4. Divide con StratifiedShuffleSplit (random_state=42).
      5. Entrena KMeans solo sobre el conjunto de entrenamiento.

    Retorna:
      (X_scaled_train, X_scaled_test, y_train, y_test, kmeans_model)
      donde X_scaled_train y X_scaled_test son numpy arrays.
    """

    df_clean = df.copy()

    # ── Paso 1: Eliminar filas con cluster nulo ──────────────────────────────
    df_clean = df_clean.dropna(subset=["cluster"]).copy()

    # ── Paso 2: Imputar income por grupo de age (rangos de 10 años) ──────────
    df_clean["age_group"] = (df_clean["age"] // 10) * 10
    df_clean["income"] = df_clean.groupby("age_group")["income"].transform(
        lambda x: x.fillna(x.mean())
    )
    # Edge case: grupo entero con NaN → media global
    if df_clean["income"].isna().any():
        df_clean["income"] = df_clean["income"].fillna(df_clean["income"].mean())

    # ── Paso 3: Escalar con RobustScaler ─────────────────────────────────────
    feature_cols = ["age", "income", "purchase_frequency", "cltv"]
    scaler = RobustScaler()
    X = scaler.fit_transform(df_clean[feature_cols])   # numpy array
    y = df_clean["cluster"].to_numpy()

    # ── Paso 4: StratifiedShuffleSplit ───────────────────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))

    # ── Arrays directos, igual que el generador ───────────────────────────────
    X_scaled_train = X[train_idx]
    X_scaled_test  = X[test_idx]
    y_train = y[train_idx]
    y_test  = y[test_idx]

    # ── Paso 5: Entrenar KMeans solo sobre train ──────────────────────────────
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_model.fit(X_scaled_train)

    return X_scaled_train, X_scaled_test, y_train, y_test, kmeans_model


# ============================================================
# USE CASE GENERATOR (copiado tal cual)
# ============================================================
def generar_caso_de_uso_generar_segmentos():
    test_size  = random.choice([0.2, 0.25, 0.3])
    n_clusters = random.choice([2, 3, 4, 5])

    n      = random.randint(40, 80)
    age    = np.random.randint(18, 70, size=n)
    income = np.random.normal(loc=30000, scale=10000, size=n)
    for i in range(n):
        if random.random() < 0.1:
            income[i] = None

    purchase_frequency = np.random.randint(1, 20, size=n)
    cltv    = np.random.uniform(100, 10000, size=n)
    cluster = [random.choice([0, 1, 2, 3, None]) for _ in range(n)]

    df = pd.DataFrame({
        "age":                age,
        "income":             income,
        "purchase_frequency": purchase_frequency,
        "cltv":               cltv,
        "cluster":            cluster
    })

    input_dict = {"df": df, "test_size": test_size, "n_clusters": n_clusters}

    # Output esperado (igual que el generador original)
    df_clean = df.dropna(subset=["cluster"]).copy()
    df_clean["age_group"] = (df_clean["age"] // 10) * 10
    df_clean["income"] = df_clean.groupby("age_group")["income"].transform(
        lambda x: x.fillna(x.mean())
    )
    if df_clean["income"].isna().any():
        df_clean["income"] = df_clean["income"].fillna(df_clean["income"].mean())

    scaler = RobustScaler()
    X = scaler.fit_transform(df_clean[["age", "income", "purchase_frequency", "cltv"]])
    y = df_clean["cluster"].to_numpy()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))

    X_scaled_train_exp = X[train_idx]
    X_scaled_test_exp  = X[test_idx]
    y_train_exp = y[train_idx]
    y_test_exp  = y[test_idx]
    kmeans_model_exp = f"KMeans(n_clusters={n_clusters}) (simulado)"

    output = (X_scaled_train_exp, X_scaled_test_exp, y_train_exp, y_test_exp, kmeans_model_exp)
    return input_dict, output


# ============================================================
# VALIDACIÓN AUTOMÁTICA
# ============================================================
def validar_solucion(n_casos=200, tolerancia=1e-9):
    errores = 0

    for i in range(n_casos):
        inp, (Xtr_exp, Xte_exp, ytr_exp, yte_exp, _) = \
            generar_caso_de_uso_generar_segmentos()

        try:
            Xtr, Xte, ytr, yte, km = generar_segmentos(
                inp["df"], inp["test_size"], inp["n_clusters"]
            )
        except Exception as e:
            print(f"[Caso {i+1}] ❌ Excepción: {e}")
            errores += 1
            continue

        ok = True

        # ── Verificar que X_scaled son numpy arrays ──────────────────────────
        if not isinstance(Xtr, np.ndarray) or not isinstance(Xte, np.ndarray):
            print(f"[Caso {i+1}] ❌ X_scaled deben ser numpy arrays, "
                  f"obtenido: {type(Xtr).__name__}, {type(Xte).__name__}")
            errores += 1
            continue

        # ── Verificar shapes ─────────────────────────────────────────────────
        if Xtr.shape != Xtr_exp.shape:
            print(f"[Caso {i+1}] ❌ Shape X_train: {Xtr.shape} vs {Xtr_exp.shape}")
            ok = False
        if Xte.shape != Xte_exp.shape:
            print(f"[Caso {i+1}] ❌ Shape X_test: {Xte.shape} vs {Xte_exp.shape}")
            ok = False

        # ── Verificar valores de X ───────────────────────────────────────────
        if ok and not np.allclose(Xtr, Xtr_exp, atol=tolerancia):
            print(f"[Caso {i+1}] ❌ Valores X_train no coinciden. "
                  f"Max diff: {np.abs(Xtr - Xtr_exp).max():.2e}")
            ok = False
        if ok and not np.allclose(Xte, Xte_exp, atol=tolerancia):
            print(f"[Caso {i+1}] ❌ Valores X_test no coinciden. "
                  f"Max diff: {np.abs(Xte - Xte_exp).max():.2e}")
            ok = False

        # ── Verificar y ──────────────────────────────────────────────────────
        if ok and not np.array_equal(ytr, ytr_exp):
            print(f"[Caso {i+1}] ❌ y_train no coincide")
            ok = False
        if ok and not np.array_equal(yte, yte_exp):
            print(f"[Caso {i+1}] ❌ y_test no coincide")
            ok = False

        # ── Verificar KMeans ─────────────────────────────────────────────────
        if ok:
            if not isinstance(km, KMeans):
                print(f"[Caso {i+1}] ❌ kmeans_model no es instancia de KMeans")
                ok = False
            elif km.n_clusters != inp["n_clusters"]:
                print(f"[Caso {i+1}] ❌ n_clusters: {km.n_clusters} vs {inp['n_clusters']}")
                ok = False

        if not ok:
            errores += 1

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

    inp, (Xtr_exp, Xte_exp, ytr_exp, yte_exp, _) = \
        generar_caso_de_uso_generar_segmentos()

    print("── Input DataFrame (primeras filas) ──")
    print(inp["df"].head(8))
    print(f"\ntest_size={inp['test_size']} | n_clusters={inp['n_clusters']}")
    print(f"NaNs en income  : {inp['df']['income'].isna().sum()}")
    print(f"NaNs en cluster : {inp['df']['cluster'].isna().sum()}")

    Xtr, Xte, ytr, yte, km = generar_segmentos(
        inp["df"], inp["test_size"], inp["n_clusters"]
    )

    print(f"\n── Resultados ──")
    print(f"X_train : shape={Xtr.shape} | tipo={type(Xtr).__name__}")
    print(f"X_test  : shape={Xte.shape} | tipo={type(Xte).__name__}")
    print(f"y_train : shape={ytr.shape}")
    print(f"y_test  : shape={yte.shape}")
    print(f"KMeans  : n_clusters={km.n_clusters} | inertia={km.inertia_:.4f}")

    print(f"\n── Coincide con esperado ──")
    print(f"  X_train : {np.allclose(Xtr, Xtr_exp)}")
    print(f"  X_test  : {np.allclose(Xte, Xte_exp)}")
    print(f"  y_train : {np.array_equal(ytr, ytr_exp)}")
    print(f"  y_test  : {np.array_equal(yte, yte_exp)}")

    print("\n── Validación con 200 casos aleatorios ──")
    validar_solucion(n_casos=200)
