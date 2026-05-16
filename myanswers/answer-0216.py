import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans

def generar_segmentos(df, test_size, n_clusters):
    """
    Procesa un DataFrame para segmentación de clientes, imputando nulos,
    escalando características, dividiendo los datos estratificadamente
    y entrenando un modelo KMeans.
    """
    # 1. Eliminar filas con cluster nulo
    df_clean = df.dropna(subset=['cluster']).copy()
    
    # 2. Imputar 'income' con la media por grupo de 'age' (rangos de 10 años)
    # Creamos una columna temporal para agrupar por décadas
    df_clean['age_group'] = (df_clean['age'] // 10) * 10
    df_clean['income'] = df_clean.groupby('age_group')['income'].transform(
        lambda x: x.fillna(x.mean())
    )
    
    # (Caso borde preventivo) Si todo un grupo de edad tenía nulos, 
    # imputamos los sobrantes con la media general
    if df_clean['income'].isnull().any():
        df_clean['income'] = df_clean['income'].fillna(df_clean['income'].mean())
        
    # Seleccionar características base y variable objetivo
    features = ['age', 'income', 'purchase_frequency', 'cltv']
    X = df_clean[features]
    y = df_clean['cluster'].astype(int) # Convertimos a entero para la estratificación
    
    # 3. Dividir datos usando StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    # 4. Escalar todas las características con RobustScaler
    scaler = RobustScaler()
    
    # Es fundamental hacer el fit SOLO en train para evitar que el modelo
    # obtenga información del test set por adelantado (Data Leakage)
    X_scaled_train_arr = scaler.fit_transform(X_train)
    X_scaled_test_arr = scaler.transform(X_test)
    
    # Convertir a DataFrames para cumplir la condición de la misión
    X_scaled_train = pd.DataFrame(X_scaled_train_arr, columns=features, index=X_train.index)
    X_scaled_test = pd.DataFrame(X_scaled_test_arr, columns=features, index=X_test.index)
    
    # 5. Entrenar el modelo KMeans
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans_model.fit(X_scaled_train)
    
    return X_scaled_train, X_scaled_test, y_train, y_test, kmeans_model
