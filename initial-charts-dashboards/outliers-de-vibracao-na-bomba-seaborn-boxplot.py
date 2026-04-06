import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

bombas = ['B1', 'B2', 'B3', 'B4']
leituras = []
for b in bombas:
    mu = np.random.uniform(0.5, 0.8)
    leituras.extend(np.random.normal(mu, 0.1, 100))
    # Injetando uma falha na B3
    if b == 'B3': leituras.extend([2.5, 2.8, 3.1]) 
    else: leituras.extend([mu, mu])

df_vibra = pd.DataFrame({'Bomba': np.repeat(bombas, 102 if 'B3' else 102), 'Vibracao': leituras[:408]})

sns.boxplot(x='Bomba', y='Vibracao', data=df_vibra, palette="Set2")
plt.title("Diagnóstico de Vibração de Bombas BCS")
plt.show()
