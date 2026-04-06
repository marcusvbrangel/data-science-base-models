import matplotlib.pyplot as plt
import numpy as np

# Simulando a descida do erro (Gradiente Descendente)
epocas = np.arange(0, 100)
erro = 100 * np.exp(-0.05 * epocas) + np.random.normal(0, 2, 100)

plt.figure(figsize=(8, 4))
plt.plot(epocas, erro, color='red', label='Função de Custo (Erro)')
plt.fill_between(epocas, erro, color='red', alpha=0.1)
plt.title("A Descida do Gradiente: O Modelo Aprendendo")
plt.xlabel("Iterações (Treino)")
plt.ylabel("Magnitude do Erro")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
