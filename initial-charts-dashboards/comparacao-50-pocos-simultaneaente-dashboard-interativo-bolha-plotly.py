import pandas as pd
import numpy as np
import plotly.express as px

df_bolha = pd.DataFrame({
    'Pressao': np.random.uniform(2000, 4000, 50),
    'Vazao': np.random.uniform(500, 8000, 50),
    'Custo': np.random.uniform(10, 100, 50),
    'Fluido': np.random.choice(['Leve', 'Pesado'], 50)
})

fig = px.scatter(df_bolha, x="Pressao", y="Vazao", size="Custo", color="Fluido",
                 hover_name="Fluido", log_x=False, size_max=40, title="Painel Estratégico de Poços")
fig.show()
