import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pressao_sensores = np.random.normal(3000, 150, 1000)

plt.figure(figsize=(8, 4))
sns.histplot(pressao_sensores, kde=True, color="purple")
plt.axvline(3000, color='k', linestyle='--', label='Média Nominal')
plt.title("Distribuição da Pressão: Detectando a Normalidade")
plt.legend()
plt.show()
