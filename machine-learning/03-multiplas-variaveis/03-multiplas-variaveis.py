import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.stats as stats

# =========================
# 1. DADOS (IMÓVEIS)
# =========================
np.random.seed(42)

n_samples = 120

size = np.random.uniform(50, 200, n_samples)        # m²
rooms = np.random.randint(1, 5, n_samples)          # quartos
age = np.random.uniform(0, 30, n_samples)           # idade
distance = np.random.uniform(1, 20, n_samples)      # distância centro

X = np.column_stack((size, rooms, age, distance))

# preço real (função oculta)
y = (
    3000 * size +
    10000 * rooms -
    2000 * age -
    5000 * distance +
    np.random.randn(n_samples) * 20000
)

feature_names = ["size", "rooms", "age", "distance"]

# =========================
# 2. NORMALIZAÇÃO
# =========================
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# =========================
# 3. HIPERPARÂMETROS
# =========================
lr = 0.01
epochs = 120

# =========================
# 4. INICIALIZAÇÃO
# =========================
n_features = X.shape[1]
W = np.zeros(n_features)
b = 0

loss_history = []
pred_history = []

# =========================
# 5. FUNÇÃO DE CUSTO
# =========================
def compute_loss(y, y_pred):
    return np.mean((y - y_pred) ** 2)

# =========================
# 6. GRADIENT DESCENT
# =========================
for epoch in range(epochs):

    y_pred = X_norm @ W + b
    error = y - y_pred

    dW = -2 * (X_norm.T @ error) / n_samples
    db = -2 * np.mean(error)

    W -= lr * dW
    b -= lr * db

    loss = compute_loss(y, y_pred)

    loss_history.append(loss)
    pred_history.append(y_pred.copy())

# =========================
# 7. RESULTADOS
# =========================
y_pred = X_norm @ W + b
residuals = y - y_pred

# =========================
# 8. MÉTRICAS
# =========================
def r2_score(y, y_pred):
    ss_total = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - y_pred)**2)
    return 1 - (ss_res / ss_total)

def mae(y, y_pred):
    return np.mean(np.abs(y - y_pred))

def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred)**2))

print("\n===== MÉTRICAS =====")
print("R²:", r2_score(y, y_pred))
print("MAE:", mae(y, y_pred))
print("RMSE:", rmse(y, y_pred))

# =========================
# 9. PLOTS PRINCIPAIS
# =========================
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Real vs Predito
axs[0,0].scatter(y, y_pred)
axs[0,0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
axs[0,0].set_title("Real vs Predicted")

# Resíduos vs Predição
axs[0,1].scatter(y_pred, residuals)
axs[0,1].axhline(0)
axs[0,1].set_title("Residuals vs Prediction")

# Histograma resíduos
axs[1,0].hist(residuals, bins=20)
axs[1,0].set_title("Residual Distribution")

# Loss
axs[1,1].plot(loss_history)
axs[1,1].set_title("Loss Convergence")

plt.tight_layout()
plt.show()

# =========================
# 10. OUTLIERS
# =========================
threshold = 2 * np.std(residuals)
outliers = np.where(np.abs(residuals) > threshold)

print("\nOutliers encontrados:", len(outliers[0]))

plt.scatter(y_pred, residuals)
plt.scatter(y_pred[outliers], residuals[outliers], marker='x')
plt.axhline(0)
plt.title("Outliers Detection")
plt.show()

# =========================
# 11. QQ PLOT
# =========================
plt.figure(figsize=(6,6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot - Residuals")
plt.show()

# =========================
# 12. FEATURE IMPORTANCE
# =========================
importance = np.abs(W)

print("\n===== FEATURE IMPORTANCE =====")
for name, imp in zip(feature_names, importance):
    print(f"{name}: {imp}")

plt.bar(feature_names, importance)
plt.title("Feature Importance")
plt.show()

# =========================
# 13. VIF
# =========================
def calculate_vif(X):
    n_features = X.shape[1]
    vif_values = []

    for i in range(n_features):
        X_i = X[:, i]
        X_rest = np.delete(X, i, axis=1)

        beta = np.linalg.pinv(X_rest) @ X_i
        X_i_pred = X_rest @ beta

        r2 = r2_score(X_i, X_i_pred)
        vif = 1 / (1 - r2)

        vif_values.append(vif)

    return vif_values

vif_values = calculate_vif(X_norm)

print("\n===== VIF =====")
for name, vif in zip(feature_names, vif_values):
    print(f"{name}: {vif}")

# =========================
# 14. DIAGNÓSTICO AUTOMÁTICO
# =========================
print("\n===== DIAGNÓSTICO =====")

r2 = r2_score(y, y_pred)

if r2 > 0.8:
    print("Modelo muito bom")
elif r2 > 0.6:
    print("Modelo razoável")
else:
    print("Modelo fraco")

for name, vif in zip(feature_names, vif_values):
    if vif > 10:
        print(f"ALERTA: multicolinearidade alta em {name}")

# =========================
# 15. ANIMAÇÃO (EVOLUÇÃO)
# =========================
fig, ax = plt.subplots()

scatter = ax.scatter(y, pred_history[0])
line, = ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')

def update(frame):
    scatter.set_offsets(np.c_[y, pred_history[frame]])
    ax.set_title(f"Epoch {frame}")
    return scatter,

ani = FuncAnimation(
    fig,
    update,
    frames=len(pred_history),
    interval=200,
    repeat=False
)

plt.show()
