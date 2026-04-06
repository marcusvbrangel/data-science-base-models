import matplotlib.pyplot as plt
import numpy as np

meses = np.arange(1, 13)
pressao = np.array([3500, 3450, 3380, 3310, 3250, 3180, 3100, 3020, 2950, 2880, 2810, 2750])

plt.figure(figsize=(10, 5))
plt.plot(meses, pressao, marker='o', linestyle='--', color='darkblue', label='Pressão de Fundo (PSI)')
plt.title("Declínio de Pressão Anual - Poço P-54")
plt.xlabel("Meses de Operação")
plt.ylabel("Pressão (PSI)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
