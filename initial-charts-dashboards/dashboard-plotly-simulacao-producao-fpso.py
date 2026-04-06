import plotly.express as px
import pandas as pd
import numpy as np

# Simulando produção de uma FPSO
df_prod = pd.DataFrame({
    'Data': pd.date_range(start='2024-01-01', periods=100),
    'Vazao_bpd': 50000 + np.cumsum(np.random.randn(100) * 1000),
    'Pressao': 3000 - (np.arange(100) * 5)
})

fig = px.line(df_prod, x='Data', y='Vazao_bpd', title='Monitoramento em Tempo Real - FPSO Bacia de Campos',
              hover_data=['Pressao'], markers=True)
fig.show()
