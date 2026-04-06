import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'Vazao': np.random.rand(100),
    'Pressao': np.random.rand(100),
    'Temp': np.random.rand(100),
    'RGO': np.random.rand(100),
    'BSW': np.random.rand(100) # % de Água
}
df_sensores = pd.DataFrame(data)

plt.figure(figsize=(8, 6))
sns.heatmap(df_sensores.corr(), annot=True, cmap='RdYlGn', center=0)
plt.title("Matriz de Correlação dos Sensores de Fluxo")
plt.show()
