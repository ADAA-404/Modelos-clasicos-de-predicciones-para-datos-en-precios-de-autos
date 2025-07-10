# Modelos-clasicos-de-predicciones-para-datos-en-precios-de-autos
Este proyecto tiene como objetivo principal la práctica y aplicación de técnicas de análisis de datos y modelos de regresión para predecir precios de automóviles. Utiliza un conjunto de datos real para explorar diferentes modelos predictivos, como la regresión lineal simple, la regresión lineal múltiple y la regresión polinomial.

# Consideraciones en Instalación
Si usamos mamba:

!mamba install pandas==1.3.3 -y

!mamba install numpy==1.21.2 -y

!mamba install scikit-learn==0.20.1 -y

!mamba install matplotlib -y

!mamba install seaborn -y


Alternativamente, si usamos pip:

pip install pandas==1.3.3

pip install numpy==1.21.2

pip install scikit-learn==0.20.1

pip install matplotlib

pip install seaborn


En esta ocasion el codigo se escribio en Jupyter Notebook para Python.


## Tecnologias usadas
- pandas: Para manipulación y análisis de datos.
- numpy: Para operaciones numéricas eficientes.
- scikit-learn (sklearn): Para la implementación de modelos de regresión lineal, transformación de características (PolynomialFeatures, StandardScaler), y métricas de evaluación (mean_squared_error, r2_score).
- matplotlib: Para la creación de gráficos estáticos.
- seaborn: Para la visualización de datos estadísticos más atractiva.

## Ejemplos de uso
Este código explora y demuestra el uso de diferentes modelos de regresión para predecir el precio de los automóviles. Veamos cómo se detalla y cómo se aplica cada modelo:
 1. Carga y Exploración Inicial de Datos: comenzamos cargando un conjunto de datos de automóviles (por parte IBM en este caso) y verificamos las primeras filas.
 2. Regresión Lineal Simple: se entrena un modelo de regresión lineal simple utilizando la variable 'highway-mpg' para predecir el 'price'. Se muestran los valores de intercepción y coeficiente.
 3. Regresión Lineal Múltiple: se entrena un modelo de regresión lineal múltiple utilizando 'horsepower', 'curb-weight', 'engine-size' y 'highway-mpg' para predecir el 'price'.
 4. Visualización de Modelos: se utilizan seaborn y matplotlib para visualizar los resultados de la regresión, incluyendo gráficos de dispersión con línea de regresión y gráficos residuales.
 5. Regresión Polinomial: se muestra cómo aplicar una regresión polinomial (de tercer orden en este caso) y se visualiza el ajuste. También se muestra el uso de PolynomialFeatures para transformar datos.
 6. Pipelines para Preprocesamiento y Modelado: se utiliza Pipeline de sklearn para encadenar operaciones como escalado de datos y transformación polinomial antes de aplicar el modelo de regresión.
 7. Evaluación del Modelo: Se calculan métricas como R^2 (coeficiente de determinación) y MSE (Error Cuadrático Medio) para evaluar el rendimiento de los diferentes modelos.

## Contribuciones
Si te interesa contribuir a este proyecto o usarlo independiente, considera:
- Hacer un "fork" del repositorio.
- Crear una nueva rama (git checkout -b feature/nueva-caracteristica).
- Realizar tus cambios y "commitearlos" (git commit -am 'Agregar nueva característica').
- Subir tus cambios a la rama (git push origin feature/nueva-caracteristica).
- Abrir un "Pull Request".

## Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE (si aplica) para más detalles.
