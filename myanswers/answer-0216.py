import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans

def generar_segmentos(df, test_size, n_clusters):
    # 1. Eliminar filas con cluster nulo
    df_clean = df.dropna(subset=["cluster"]).copy()
    
    # 2. Imputar 'income' con la media por grupo de 'age' (rangos de 10 años)
    df_clean["age_group"] = (df_clean["age"] // 10) * 10
    df_clean["income"] = df_clean.groupby("age_group")["income"].transform(lambda x: x.fillna(x.mean()))
    
    if df_clean["income"].isnull().any():
        df_clean["income"] = df_clean["income"].fillna(df_clean["income"].mean())
        
    # 3. Extraer estrictamente como NumPy arrays para evitar que Scikit-Learn devuelva DataFrames
    features = ["age", "income", "purchase_frequency", "cltv"]
    X_raw = df_clean[features].to_numpy() 
    y = df_clean["cluster"].to_numpy()
    
    # 4. Escalar todas las características
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Redundancia de seguridad: asegurar que el tipo sea ndarray
    X_scaled = np.asarray(X_scaled)
    
    # 5. Dividir datos usando StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(sss.split(X_scaled, y))
    
    X_scaled_train = X_scaled[train_idx]
    X_scaled_test = X_scaled[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    # 6. Entrenar el modelo KMeans
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans_model.fit(X_scaled_train)
    
    return X_scaled_train, X_scaled_test, y_train, y_test, kmeans_model
