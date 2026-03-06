import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random

def generar_caso_de_uso_ensamble_votacion_suave():
    # 1. Configuración aleatoria controlada
    n_rows = random.randint(150, 250)
    
    # 2. Generar datos de clasificación con superposición (Gaussinas)
    # Clase 0
    X_class0 = np.random.normal(loc=[2.0, 2.0], scale=[1.5, 1.5], size=(n_rows // 2, 2))
    y_class0 = np.zeros(n_rows // 2)
    
    # Clase 1 (centros cercanos para que haya solapamiento y el ensamble tenga que trabajar)
    X_class1 = np.random.normal(loc=[4.0, 4.0], scale=[1.5, 1.5], size=(n_rows - (n_rows // 2), 2))
    y_class1 = np.ones(n_rows - (n_rows // 2))
    
    X_data = np.vstack((X_class0, X_class1))
    y_data = np.concatenate((y_class0, y_class1))
    
    # Mezclar los datos
    indices = np.arange(n_rows)
    np.random.shuffle(indices)
    X_data = X_data[indices]
    y_data = y_data[indices].astype(int)
    
    feature_cols = ['sensor_A', 'sensor_B']
    df = pd.DataFrame(X_data, columns=feature_cols)
    df['clase_objetivo'] = y_data
    
    # 3. INPUT
    input_data = {
        'df': df.copy(),
        'feature_cols': feature_cols,
        'target_col': 'clase_objetivo'
    }
    
    # 4. OUTPUT ESPERADO
    X = df[feature_cols]
    y_true = df['clase_objetivo']
    
    rf = RandomForestClassifier(random_state=42)
    svc = SVC(probability=True, random_state=42)
    ensamble = VotingClassifier(estimators=[('rf', rf), ('svc', svc)], voting='soft')
    
    ensamble.fit(X, y_true)
    acc = accuracy_score(y_true, ensamble.predict(X))
    
    output_expected = (ensamble, acc)
    
    return input_data, output_expected
