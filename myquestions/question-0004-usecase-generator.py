import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import random

def generar_caso_de_uso_clasificacion_umbral_personalizado():
    # 1. Configuración aleatoria
    n_rows = random.randint(200, 400)
    # Umbral bajo para simular precaución médica (aumentar recall)
    umbral_custom = round(random.uniform(0.25, 0.40), 2) 
    
    # 2. Generar datos: Biométricas de pacientes
    # Pacientes sanos (Clase 0)
    sanos_f1 = np.random.normal(120, 15, int(n_rows * 0.7)) # Presión arterial normal
    sanos_f2 = np.random.normal(80, 10, int(n_rows * 0.7))  # Glucosa normal
    
    # Pacientes en riesgo (Clase 1 - minoría)
    riesgo_f1 = np.random.normal(150, 20, int(n_rows * 0.3)) # Presión alta
    riesgo_f2 = np.random.normal(110, 15, int(n_rows * 0.3)) # Glucosa alta
    
    X_f1 = np.concatenate((sanos_f1, riesgo_f1))
    X_f2 = np.concatenate((sanos_f2, riesgo_f2))
    y = np.concatenate((np.zeros(len(sanos_f1)), np.ones(len(riesgo_f1))))
    
    # Empaquetar y barajar
    df = pd.DataFrame({'presion_arterial': X_f1, 'glucosa': X_f2, 'complicacion': y.astype(int)})
    df = df.sample(frac=1, random_state=random.randint(0, 1000)).reset_index(drop=True)
    
    target_col = 'complicacion'
    
    # 3. INPUT
    input_data = {
        'df': df.copy(),
        'target_col': target_col,
        'umbral': umbral_custom
    }
    
    # 4. OUTPUT ESPERADO
    X = df[['presion_arterial', 'glucosa']]
    y_true = df[target_col]
    
    nb = GaussianNB()
    nb.fit(X, y_true)
    
    # Extraer las probabilidades de que sea clase 1
    probabilidades = nb.predict_proba(X)[:, 1]
    
    # Aplicar el umbral personalizado
    predicciones = np.where(probabilidades >= umbral_custom, 1, 0)
    
    output_expected = (probabilidades, predicciones)
    
    return input_data, output_expected
