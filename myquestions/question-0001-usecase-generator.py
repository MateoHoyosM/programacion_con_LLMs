import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import random

def generar_caso_de_uso_entrenar_evaluar_arbol():
    # 1. Configuración aleatoria (no trivial: suficientes datos para capturar el patrón)
    n_rows = random.randint(200, 500)
    max_depth = random.randint(3, 7)
    
    # 2. Generación de datos con patrón no lineal
    X1 = np.random.uniform(-5, 5, n_rows)
    X2 = np.random.uniform(-5, 5, n_rows)
    X3 = np.random.uniform(0, 10, n_rows) # Variable trampa (casi no afecta)
    
    # y = sen(X1) + cos(X2) + algo de ruido
    y = np.sin(X1) * 3 + np.cos(X2) * 2 + (X3 * 0.1) + np.random.normal(0, 0.5, n_rows)
    
    df = pd.DataFrame({'var_1': X1, 'var_2': X2, 'var_ruido': X3, 'objetivo': y})
    feature_cols = ['var_1', 'var_2', 'var_ruido']
    target_col = 'objetivo'
    
    # 3. INPUT
    input_data = {
        'df': df.copy(),
        'feature_cols': feature_cols,
        'target_col': target_col,
        'max_depth': max_depth
    }
    
    # 4. OUTPUT ESPERADO
    X = df[feature_cols]
    y_true = df[target_col]
    
    modelo = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    modelo.fit(X, y_true)
    y_pred = modelo.predict(X)
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    output_expected = (mse, r2)
    
    return input_data, output_expected
