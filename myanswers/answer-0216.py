import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans


def generar_segmentos(df, test_size, n_clusters):

    df_clean = df.copy()

    # ── Paso 1: Eliminar filas con cluster nulo ──────────────────────────────
    df_clean = df_clean.dropna(subset=["cluster"]).copy()

    # ── Paso 2: Imputar income por grupo de age (rangos de 10 años) ──────────
    df_clean["age_group"] = (df_clean["age"] // 10) * 10
    df_clean["income"] = df_clean.groupby("age_group")["income"].transform(
        lambda x: x.fillna(x.mean())
    )
    if df_clean["income"].isna().any():
        df_clean["income"] = df_clean["income"].fillna(df_clean["income"].mean())

    # ── Paso 3: Escalar con RobustScaler ─────────────────────────────────────
    scaler = RobustScaler()
    X = scaler.fit_transform(df_clean[["age", "income", "purchase_frequency", "cltv"]])
    y = df_clean["cluster"].to_numpy()

    # ── Paso 4: StratifiedShuffleSplit ───────────────────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))

    X_scaled_train = X[train_idx]   # numpy array
    X_scaled_test  = X[test_idx]    # numpy array
    y_train = y[train_idx]
    y_test  = y[test_idx]

    # ── Paso 5: KMeans real con n_init=10 y random_state=42 ──────────────────
    kmeans_model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans_model.fit(X_scaled_train)

    return X_scaled_train, X_scaled_test, y_train, y_test, kmeans_model
