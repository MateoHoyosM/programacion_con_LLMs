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
    
    # Prevenir errores si un grupo de edad entero fuera nulo
    if df_clean["income"].isnull().any():
        df_clean["income"] = df_clean["income"].fillna(df_clean["income"].mean())
        
    # 3. Escalar todas las características (Coincidiendo con el orden del test)
    features = ["age", "income", "purchase_frequency", "cltv"]
    scaler = RobustScaler()
    
    # scaler.fit_transform devuelve un array de NumPy directamente
    X = scaler.fit_transform(df_clean[features]) 
    y = df_clean["cluster"].to_numpy() # Convertimos la etiqueta a array también
    
    # 4. Dividir datos usando StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_scaled_train, X_scaled_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 5. Entrenar el modelo KMeans usando solo el conjunto de entrenamiento
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans_model.fit(X_scaled_train)
    
    return X_scaled_train, X_scaled_test, y_train, y_test, kmeans_model
