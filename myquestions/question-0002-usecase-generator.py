import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
import random

def generar_caso_de_uso_seleccionar_features_polinomiales():
    # 1. Configuración
    n_rows = random.randint(150, 300)
    degree = random.randint(2, 3)
    alpha = round(random.uniform(0.5, 1.5), 2) # Penalización fuerte para obligar ceros
    
    # 2. Generar datos
    X1 = np.random.uniform(-2, 2, n_rows)
    X2 = np.random.uniform(-2, 2, n_rows)
    X3 = np.random.normal(0, 1, n_rows) # Ruido 1
    X4 = np.random.normal(0, 1, n_rows) # Ruido 2
    
    # El target depende de X1^2 y de la interacción X1*X2. Las variables X3 y X4 no sirven.
    y = 4.5 * (X1 ** 2) - 2.0 * (X1 * X2) + np.random.normal(0, 0.2, n_rows)
    
    df = pd.DataFrame({'f1': X1, 'f2': X2, 'ruido1': X3, 'ruido2': X4, 'target': y})
    feature_cols = ['f1', 'f2', 'ruido1', 'ruido2']
    
    # 3. INPUT
    input_data = {
        'df': df.copy(),
        'feature_cols': feature_cols,
        'target_col': 'target',
        'degree': degree,
        'alpha': alpha
    }
    
    # 4. OUTPUT ESPERADO
    X = df[feature_cols]
    y_true = df['target']
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X_poly, y_true)
    
    # Extraer índices de coeficientes distintos de cero
    indices_importantes = np.where(lasso.coef_ != 0)[0]
    
    output_expected = indices_importantes
    
    return input_data, output_expected
