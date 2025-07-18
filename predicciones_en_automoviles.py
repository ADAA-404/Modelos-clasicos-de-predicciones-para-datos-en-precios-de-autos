"""
predicciones_en_automoviles.py

Este script realiza un análisis exploratorio de datos y aplica diversos modelos de regresión
para predecir precios de automóviles, utilizando un conjunto de datos (proporcionado por IBM).
El objetivo es demostrar diferentes técnicas clasicas de modelado predictivo y su evaluación.

Contenido:
- Carga de datos.
- Implementación y evaluación de Regresión Lineal Simple.
- Implementación y evaluación de Regresión Lineal Múltiple.
- Implementación y evaluación de Regresión Polinomial.
- Uso de Pipelines para preprocesamiento y modelado.
- Visualización de resultados de los modelos.
- Cálculo de métricas de evaluación (R^2, MSE).

Librerías principales utilizadas: pandas, numpy, scikit-learn, matplotlib, seaborn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def load_data(url: str) -> pd.DataFrame:
    """
    Carga un conjunto de datos desde una URL especificada.

    Args:
        url (str): La URL del archivo CSV a cargar.

    Returns:
        pd.DataFrame: El DataFrame de pandas con los datos cargados.
    """
    print(f"Cargando datos desde: {url}")
    df = pd.read_csv(url)
    print("Primeras 5 filas del dataset:")
    print(df.head())
    return df

def perform_simple_linear_regression(df: pd.DataFrame, feature_col: str, target_col: str) -> tuple:
    """
    Realiza una regresión lineal simple y evalúa el modelo.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        feature_col (str): Nombre de la columna de la variable independiente (característica).
        target_col (str): Nombre de la columna de la variable dependiente (objetivo).

    Returns:
        tuple: Una tupla que contiene (modelo, predicciones, R^2, MSE).
    """
    print(f"\n--- Regresión Lineal Simple: {feature_col} vs {target_col} ---")
    lm = LinearRegression()
    X = df[[feature_col]] # Variable independiente debe ser un DataFrame
    Y = df[target_col]   # Variable dependiente puede ser una Serie

    lm.fit(X, Y)
    Yhat = lm.predict(X)

    r_squared = lm.score(X, Y)
    mse = mean_squared_error(Y, Yhat)

    print(f"Coeficiente de Intercepto: {lm.intercept_:.2f}")
    print(f"Coeficiente de '{feature_col}': {lm.coef_[0]:.2f}")
    print(f"R-cuadrado: {r_squared:.4f}")
    print(f"Error Cuadrático Medio (MSE): {mse:.2f}")

    # Visualización de la regresión lineal simple
    plt.figure(figsize=(12, 8))
    sns.regplot(x=feature_col, y=target_col, data=df)
    plt.ylim(0,)
    plt.title(f'Regresión Lineal Simple: {feature_col} vs {target_col}')
    plt.xlabel(feature_col)
    plt.ylabel(target_col)
    plt.show()

    # Visualización del Residual Plot
    plt.figure(figsize=(12, 8))
    sns.residplot(x=df[feature_col], y=df[target_col])
    plt.title(f'Gráfico de Residuales para {feature_col}')
    plt.xlabel(feature_col)
    plt.ylabel('Residuales')
    plt.show()

    return lm, Yhat, r_squared, mse

def perform_multiple_linear_regression(df: pd.DataFrame, feature_cols: list, target_col: str) -> tuple:
    """
    Realiza una regresión lineal múltiple y evalúa el modelo.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        feature_cols (list): Lista de nombres de las columnas de las variables independientes.
        target_col (str): Nombre de la columna de la variable dependiente (objetivo).

    Returns:
        tuple: Una tupla que contiene (modelo, predicciones, R^2, MSE).
    """
    print(f"\n--- Regresión Lineal Múltiple: {feature_cols} vs {target_col} ---")
    lm_multi = LinearRegression()
    Z = df[feature_cols]
    Y = df[target_col]

    lm_multi.fit(Z, Y)
    Y_predict_multifit = lm_multi.predict(Z)

    r_squared = lm_multi.score(Z, Y)
    mse = mean_squared_error(Y, Y_predict_multifit)

    print(f"Coeficiente de Intercepto (Múltiple): {lm_multi.intercept_:.2f}")
    print(f"Coeficientes (Múltiple): {lm_multi.coef_}")
    print(f"R-cuadrado (Múltiple): {r_squared:.4f}")
    print(f"Error Cuadrático Medio (MSE Múltiple): {mse:.2f}")

    # Visualización: Distribución de valores reales vs predichos
    plt.figure(figsize=(12, 8))
    ax1 = sns.distplot(Y, hist=False, color="r", label="Valor Real")
    sns.distplot(Y_predict_multifit, hist=False, color="b", label="Valores Ajustados", ax=ax1)
    plt.title('Valores Reales vs Ajustados para el Precio (Regresión Múltiple)')
    plt.xlabel('Precio (en dólares)')
    plt.ylabel('Proporción de Automóviles')
    plt.legend()
    plt.show()

    return lm_multi, Y_predict_multifit, r_squared, mse

def plot_polly(model, independent_variable: np.ndarray, dependent_variable: np.ndarray, name: str):
    """
    Función auxiliar para graficar un ajuste polinomial.

    Args:
        model: El modelo polinomial generado (e.g., np.poly1d).
        independent_variable (np.ndarray): Los datos de la variable independiente.
        dependent_variable (np.ndarray): Los datos de la variable dependiente.
        name (str): Nombre de la variable independiente para la etiqueta del eje.
    """
    x_new = np.linspace(independent_variable.min(), independent_variable.max(), 100)
    y_new = model(x_new)

    plt.figure(figsize=(12, 8))
    plt.plot(independent_variable, dependent_variable, '.', label='Datos Reales')
    plt.plot(x_new, y_new, '-', label='Ajuste Polinomial')
    plt.title(f'Ajuste Polinomial para Precio ~ {name}')
    plt.xlabel(name)
    plt.ylabel('Precio de Automóviles')
    plt.legend()
    plt.grid(True)
    plt.show()

def perform_polynomial_regression(df: pd.DataFrame, feature_col: str, target_col: str, degree: int = 3) -> tuple:
    """
    Realiza una regresión polinomial y evalúa el modelo.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        feature_col (str): Nombre de la columna de la variable independiente.
        target_col (str): Nombre de la columna de la variable dependiente.
        degree (int): Grado del polinomio a ajustar.

    Returns:
        tuple: Una tupla que contiene (modelo polinomial, R^2, MSE).
    """
    print(f"\n--- Regresión Polinomial (Grado {degree}): {feature_col} vs {target_col} ---")
    x = df[feature_col].values
    y = df[target_col].values

    # Se ajusta un polinomio de n-ésimo orden usando numpy
    f = np.polyfit(x, y, degree)
    p = np.poly1d(f)
    print(f"Ecuación Polinomial:\n{p}")

    Yhat_poly = p(x)
    r_squared = r2_score(y, Yhat_poly)
    mse = mean_squared_error(y, Yhat_poly)

    print(f"R-cuadrado (Polinomial): {r_squared:.4f}")
    print(f"Error Cuadrático Medio (MSE Polinomial): {mse:.2f}")

    # Visualización del ajuste polinomial
    plot_polly(p, x, y, feature_col)

    # Demostración de PolynomialFeatures (transformación para scikit-learn)
    pr = PolynomialFeatures(degree=degree)
    X_pr = pr.fit_transform(df[[feature_col]])
    print(f"Dimensiones de X original: {df[[feature_col]].shape}")
    print(f"Dimensiones de X transformado por PolynomialFeatures (grado {degree}): {X_pr.shape}")

    return p, r_squared, mse

def use_pipeline_for_modeling(df: pd.DataFrame, feature_cols: list, target_col: str, poly_degree: int = 2) -> tuple:
    """
    Demuestra el uso de un Pipeline para preprocesamiento (escalado, transformación polinomial)
    y modelado con regresión lineal.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        feature_cols (list): Lista de nombres de las columnas de las variables independientes.
        target_col (str): Nombre de la columna de la variable dependiente.
        poly_degree (int): Grado del polinomio para PolynomialFeatures.

    Returns:
        tuple: Una tupla que contiene (modelo pipeline, predicciones).
    """
    print(f"\n--- Uso de Pipeline para Modelado ---")
    # Asegurarse de que las columnas de características sean de tipo float para StandardScaler
    Z = df[feature_cols].astype(float)
    y = df[target_col]

    # Definición del pipeline: escalado -> transformación polinomial -> modelo de regresión
    input_pipeline = [
        ('scale', StandardScaler()),
        ('polynomial', PolynomialFeatures(include_bias=False, degree=poly_degree)),
        ('model', LinearRegression())
    ]
    pipe = Pipeline(input_pipeline)
    print("Estructura del Pipeline:")
    print(pipe)

    # Entrenar el pipeline
    pipe.fit(Z, y)
    ypipe = pipe.predict(Z)

    print(f"Primeras 5 predicciones del Pipeline: {ypipe[0:5]}")

    # Evaluación del pipeline (opcional, pero buena práctica)
    r_squared_pipe = pipe.score(Z, y)
    mse_pipe = mean_squared_error(y, ypipe)
    print(f"R-cuadrado (Pipeline): {r_squared_pipe:.4f}")
    print(f"Error Cuadrático Medio (MSE Pipeline): {mse_pipe:.2f}")

    return pipe, ypipe

def interpret_predictions(model, feature_col: str, target_col: str):
    """
    Demuestra cómo interpretar las predicciones de un modelo lineal.
    Genera nuevas entradas y visualiza las predicciones.

    Args:
        model: El modelo de regresión lineal entrenado.
        feature_col (str): Nombre de la columna de la característica usada para el modelo.
        target_col (str): Nombre de la columna del objetivo.
    """
    print(f"\n--- Interpretación de Predicciones para '{feature_col}' ---")
    # Crear nuevas entradas en un rango para ver la tendencia de predicción
    new_input = np.arange(1, 100, 1).reshape(-1, 1)

    # Asegurarse de que el modelo esté ajustado con la característica correcta si no lo está
    # Esto es solo un ejemplo, en un flujo real el modelo ya estaría entrenado para la interpretación
    # model.fit(df[[feature_col]], df[target_col]) # No es necesario si el modelo ya está entrenado

    yhat_new = model.predict(new_input)

    plt.figure(figsize=(10, 6))
    plt.plot(new_input, yhat_new, color='blue', linewidth=2, label='Predicción del Modelo')
    plt.title(f'Predicciones del Precio basadas en {feature_col}')
    plt.xlabel(f'{feature_col}')
    plt.ylabel(f'{target_col}')
    plt.grid(True)
    plt.legend()
    plt.show()
    print("Visualización de cómo el modelo predice el precio para nuevas entradas de la característica.")


# --- Bloque de ejecución principal ---
if __name__ == "__main__":
    # URL de la base de datos
    DATA_URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'

    # Cargar los datos
    automobile_df = load_data(DATA_URL)

    # Ejecutar y evaluar Regresión Lineal Simple
    lm_simple, yhat_simple, r2_simple, mse_simple = perform_simple_linear_regression(
        automobile_df, 'highway-mpg', 'price'
    )

    # Ejecutar y evaluar Regresión Lineal Múltiple
    features_multi = ['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']
    lm_multi, yhat_multi, r2_multi, mse_multi = perform_multiple_linear_regression(
        automobile_df, features_multi, 'price'
    )

    # Ejecutar y evaluar Regresión Polinomial
    poly_model, r2_poly, mse_poly = perform_polynomial_regression(
        automobile_df, 'highway-mpg', 'price', degree=3
    )

    # Demostrar el uso de Pipeline
    pipeline_model, ypipe_predictions = use_pipeline_for_modeling(
        automobile_df, features_multi, 'price', poly_degree=2
    )

    # Interpretar predicciones con el modelo lineal simple (como ejemplo)
    interpret_predictions(lm_simple, 'highway-mpg', 'price')

    print("\n--- Análisis de Regresión Completado ---")
    print("Revisa los gráficos generados y las métricas en la consola para los resultados.")