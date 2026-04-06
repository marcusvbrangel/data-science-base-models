
'''
The objective of this code is to implement a simple linear regression from scratch. In other words, 
we are calculating the line that best fits the data using only mathematical formulas. 
What it ultimately solves is predicting the value of y based on x, using a line that we calculate manually.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('machine-learning/dados.csv')

# select the columns you want (for example X and Y)
# X = data['X'].values # independent variable
# Y = data['Y'].values # dependent variable

# simulating data with noise, so that this no more will be linear
np.random.seed(42)
X = np.array([1,2,3,4,5])
Y = 2*X + np.random.randn(5)
print(np.random.randn(5)) # [-0.23413696  1.57921282  0.76743473 -0.46947439  0.54256004]

# calculate de mean of X and Y
X_mean = np.mean(X) # 3.0
Y_mean = np.mean(Y) # 6.0

# calculate the angular coefficient (m) manually or calculate the slope
# what is the best value (m) that minimize the error
# X             ->  [1 2 3 4 5]
# X_mean        ->  3.0
# (X - X_mean)  ->  [-2. -1.  0.  1.  2.]
# numerador     ->  20.0
# denominador   ->  10.0
# m             ->  2.0
numerador = np.sum((X - X_mean) * (Y - Y_mean))
denominador = np.sum((X - X_mean) ** 2)
m = numerador / denominador  # angular coefficient or slope

# calculate the linear coefficient (b)
b = Y_mean - m * X_mean  # 0.0 

# make the prediction using the linear equation
y_pred = m * X + b

# display the real and predicts values
print("Valores reais: ", Y)
print("Valores previstos: ", y_pred)

# calculate the error
errors = Y - y_pred
# MSE
mse = np.mean(errors ** 2)
# RMSE
rmse = np.sqrt(mse)

print("Errors: ", errors)
print("MSE: ", mse)
print("RMSE", rmse)

# calcular R² (coeficiente de determinação)
# quanto da variação dos dados o modelo consegue explicar?
# SS_res (erro)
ss_res = np.sum((Y - y_pred) ** 2)
# SS_tot (variacao total)
ss_tot = np.sum((Y - Y_mean) ** 2)
# R²
r2 = 1 - (ss_res / ss_tot)
print("R²: ", r2)

# plot the chart
# real values
plt.figure()
plt.scatter(X, Y, label="Real Values")
# predicted values
plt.plot(X, y_pred, label="Predicted values")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Real vs Predicted")
plt.legend()
# plt.show()

# plotar os residuos 
# residuos
residuals = Y - y_pred
# plot
plt.figure()
plt.scatter(X, residuals)
# linha horizontal no zero (super importante)
plt.axhline(0)
plt.xlabel("X")
plt.ylabel("Residuals")
plt.legend("Residual Plot")
plt.legend()
plt.show()

# detectar outliers
# Método 1 — Z-score (rápido e direto)
mean_res = np.mean(residuals)
std_res = np.std(residuals)
z_scores = (residuals - mean_res) / std_res
print("Z-scores: ", z_scores)
outliers = np.where(np.abs(z_scores) > 2)
print("Outliers (indices):", outliers)

# detectar outliers
# Método 2 — IQR (mais robusto)
Q1 = np.percentile(residuals, 25)
Q3 = np.percentile(residuals, 75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
print("Bounds:", lower, upper)
outliers = residuals[(residuals < lower) | (residuals > upper)]
print("Outliers:", outliers)

















