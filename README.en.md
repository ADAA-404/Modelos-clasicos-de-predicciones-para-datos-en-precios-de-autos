[Versión en Español](README.md)


# Classic Prediction Models for Car Prices Data 🚗
This project primarily aims to practice and apply data analysis techniques and regression models to predict car prices. It uses a real dataset to explore different predictive models, such as simple linear regression,
multiple linear regression, and polynomial regression.


## Technologies Used 🐍
- pandas: For data manipulation and analysis.
- numpy: For efficient numerical operations.
- scikit-learn (sklearn): For implementing linear regression models, feature transformations (PolynomialFeatures, StandardScaler), and evaluation metrics (mean_squared_error, r2_score).
- matplotlib: For creating static plots.
- seaborn: For more attractive statistical data visualization.

## Installation Considerations ⚙️
If you're using mamba:

!mamba install pandas==1.3.3 -y

!mamba install numpy==1.21.2 -y

!mamba install scikit-learn==0.20.1 -y

!mamba install matplotlib -y

!mamba install seaborn -y


Alternatively, if you're using pip:

pip install pandas==1.3.3

pip install numpy==1.21.2

pip install scikit-learn==0.20.1

pip install matplotlib

pip install seaborn


For this project, the code was written in Jupyter Notebook for Python.


## Usage Example 📎
This code explores and demonstrates the use of different regression models to predict car prices. Here's how each model is applied and understood:
 1. Data Loading and Initial Exploration: We start by loading a car dataset (from IBM in this case) and checking the first few rows.
 2. Simple Linear Regression: A simple linear regression model is trained using the 'highway-mpg' variable to predict 'price'. The intercept and coefficient values are displayed.
 3. Multiple Linear Regression: A multiple linear regression model is trained using 'horsepower', 'curb-weight', 'engine-size', and 'highway-mpg' to predict 'price'.
 4. Model Visualization: Seaborn and Matplotlib are used to visualize the regression results, including scatter plots with regression lines and residual plots.
 5. Polynomial Regression: This section demonstrates how to apply polynomial regression (third-order in this case) and visualizes the fit. It also shows the use of PolynomialFeatures for data transformation.
 6. Pipelines for Preprocessing and Modeling: Scikit-learn's Pipeline is used to chain operations like data scaling and polynomial transformation before applying the regression model.
 7. Model Evaluation: Metrics such as R^2 (coefficient of determination) and MSE (Mean Squared Error) are calculated to evaluate the performance of the different models.

## Contributions 🖨️
If you're interested in contributing to this project or using it independently, consider:
- Forking the repository.
- Creating a new branch (git checkout -b feature/new-feature).
- Making your changes and committing them (git commit -am 'Add new feature').
- Pushing your changes to the branch (git push origin feature/new-feature).
- Opening a 'Pull Request'.

## License 📜
This project is under the MIT License. Refer to the LICENSE file (if applicable) for more details.
