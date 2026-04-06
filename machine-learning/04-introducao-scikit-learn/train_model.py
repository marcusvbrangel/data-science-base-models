"""
train_model.py

Pipeline completo de Machine Learning:
- Geração de dados
- Treinamento
- Avaliação
- Comparação de modelos
- Diagnóstico estatístico
- Salvamento do modelo
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# =========================
# 1. DADOS
# =========================
np.random.seed(42)
n_samples = 120

size = np.random.uniform(50, 200, n_samples)
rooms = np.random.randint(1, 5, n_samples)
age = np.random.uniform(0, 30, n_samples)
distance = np.random.uniform(1, 20, n_samples)
garage = np.random.randint(0, 2, n_samples)

X = np.column_stack((size, rooms, age, distance, garage))

print('-------------------------------')
print('X: ', X)
print('-------------------------------')

y = (
    3000 * size +
    10000 * rooms -
    2000 * age -
    5000 * distance +
    15000 * garage +
    np.random.randn(n_samples) * 20000
)

print('-------------------------------')
print('y: ', y)
print('-------------------------------')

feature_names = ["size", "rooms", "age", "distance", "garage"]

# =========================
# 2. SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 3. PIPELINES
# =========================
"""
Cada modelo já vem com:
- Normalização
- Treinamento

Isso evita erro em produção
"""

models = {
    "Linear": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1))
    ]),
    "RandomForest": RandomForestRegressor(n_estimators=100)
}

results = {}

# =========================
# 4. TREINAMENTO E AVALIAÇÃO
# =========================
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    results[name] = {
        "model": model,
        "r2": r2,
        "rmse": rmse,
        "cv_mean": cv_scores.mean()
    }

    print(f"\n===== {name} =====")
    print("R²:", r2)
    print("RMSE:", rmse)
    print("CV R²:", cv_scores)
    print("CV Mean:", cv_scores.mean())

# =========================
# 5. MELHOR MODELO
# =========================
best_model_name = max(results, key=lambda x: results[x]["cv_mean"])
best_model = results[best_model_name]["model"]

print("\n🏆 MELHOR MODELO:", best_model_name)

# =========================
# 6. DIAGNÓSTICO
# =========================
y_pred = best_model.predict(X_test)
residuals = y_test - y_pred

plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.title("Resíduos vs Predição")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(residuals, bins=20)
plt.title("Distribuição dos Resíduos")
plt.show()

plt.figure(figsize=(6,6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot")
plt.show()

# =========================
# 7. OUTLIERS
# =========================
z_scores = np.abs(stats.zscore(residuals))
outliers = np.where(z_scores > 3)

print("\nOutliers encontrados:", len(outliers[0]))

# =========================
# 8. FEATURE IMPORTANCE
# =========================
if hasattr(best_model, "named_steps"):
    model_step = best_model.named_steps["model"]
    if hasattr(model_step, "coef_"):
        importance = np.abs(model_step.coef_)
        print("\nFeature Importance:")
        for name, imp in zip(feature_names, importance):
            print(f"{name}: {imp}")

# =========================
# 9. SALVAR MODELO
# =========================
joblib.dump(best_model, "model.pkl")

print("\n✅ Modelo salvo como model.pkl")
