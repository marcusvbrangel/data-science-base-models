import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# 1. DADOS
# =========================
np.random.seed(42)

X = np.linspace(0, 10, 50)
y = 3 * X + 4 + np.random.randn(50) * 2

# normalização
X_norm = (X - X.mean()) / X.std()

# =========================
# 2. HIPERPARÂMETROS
# =========================
lr = 0.05
epochs = 100

# =========================
# 3. INICIALIZAÇÃO
# =========================
w = 0
b = 0

w_history = []
b_history = []
loss_history = []

# =========================
# 4. FUNÇÃO DE CUSTO
# =========================
def compute_loss(y, y_pred):
    return np.mean((y - y_pred) ** 2)

# =========================
# 5. GRADIENT DESCENT
# =========================
for epoch in range(epochs):

    y_pred = w * X_norm + b

    dw = -2 * np.mean(X_norm * (y - y_pred))
    db = -2 * np.mean(y - y_pred)

    w = w - lr * dw
    b = b - lr * db

    loss = compute_loss(y, y_pred)

    w_history.append(w)
    b_history.append(b)
    loss_history.append(loss)

# =========================
# 6. PLOTS
# =========================
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# -------------------------
# (1) REGRESSÃO
# -------------------------
axs[0].scatter(X_norm, y)
line, = axs[0].plot(X_norm, w_history[0]*X_norm + b_history[0])
axs[0].set_title("Regression")

# -------------------------
# (2) LOSS
# -------------------------
axs[1].plot(loss_history)
point_loss, = axs[1].plot([0], [loss_history[0]], marker='o')
axs[1].set_title("Loss Convergence")

# -------------------------
# (3) PARÂMETROS (w, b)
# -------------------------
axs[2].set_title("Parameter Space (w, b)")

axs[2].set_xlim(min(w_history)-0.5, max(w_history)+0.5)
axs[2].set_ylim(min(b_history)-0.5, max(b_history)+0.5)

# caminho completo (sombra)
axs[2].plot(w_history, b_history, linestyle='dashed', alpha=0.3)

path, = axs[2].plot([], [])
current_point, = axs[2].plot([w_history[0]], [b_history[0]], marker='o')

# =========================
# 7. ANIMAÇÃO
# =========================
def update(frame):

    w = w_history[frame]
    b = b_history[frame]

    # --- reta
    y_pred = w * X_norm + b
    line.set_ydata(y_pred)

    # --- loss
    point_loss.set_data([frame], [loss_history[frame]])

    # --- trajetória (w, b)
    path.set_data(w_history[:frame+1], b_history[:frame+1])
    current_point.set_data([w], [b])

    axs[0].set_title(f"Epoch {frame}")

    return line, point_loss, path, current_point

# =========================
# 8. CONTROLE DE VELOCIDADE
# =========================
ani = FuncAnimation(
    fig,
    update,
    frames=len(w_history),
    interval=500,   # 🔥 controla velocidade (aumenta pra ficar mais lento)
    repeat=False
)

plt.tight_layout()
plt.show()