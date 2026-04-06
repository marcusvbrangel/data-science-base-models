import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulando dados de 100 poços (PCA Reduzido)
df_pca = pd.DataFrame({
    'PC1': np.random.randn(100),
    'PC2': np.random.randn(100),
    'Tipo': np.random.choice(['Óleo', 'Água', 'Gás'], 100)
})

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Tipo', style='Tipo', s=100)
plt.title("PCA: Agrupamento de Poços por Comportamento")
plt.show()
