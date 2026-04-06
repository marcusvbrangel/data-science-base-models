import numpy as np
import matplotlib.pyplot as plt

oleo = np.random.uniform(1000, 5000, 100)
gas = oleo * 150 + np.random.normal(0, 5000, 100)

plt.scatter(oleo, gas, alpha=0.6, edgecolors='w')
plt.xlabel("Produção de Óleo (bpd)")
plt.ylabel("Produção de Gás (m3/d)")
plt.title("Dispersão: Óleo vs Gás (Identificando o RGO)")
plt.show()
