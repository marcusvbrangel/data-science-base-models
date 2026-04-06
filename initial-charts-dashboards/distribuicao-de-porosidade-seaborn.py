import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

porosidade = np.random.normal(loc=0.22, scale=0.04, size=500) # Média 22%

sns.displot(porosidade, kde=True, color="seagreen", bins=30)
plt.axvline(np.mean(porosidade), color='red', label=f'Média: {np.mean(porosidade):.2%}')
plt.title("Histograma de Porosidade do Campo")
plt.legend()
plt.show()
