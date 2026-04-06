import plotly.graph_objects as go
import numpy as np

dias = np.arange(1, 31)
poco_A = 5000 + np.cumsum(np.random.randn(30) * 100)
poco_B = 4000 + np.cumsum(np.random.randn(30) * 150)

fig = go.Figure()
fig.add_trace(go.Scatter(x=dias, y=poco_A, mode='lines+markers', name='Poço A (Pré-Sal)'))
fig.add_trace(go.Scatter(x=dias, y=poco_B, mode='lines+markers', name='Poço B (Pós-Sal)'))
fig.update_layout(title='Monitoramento de Produção Diária (bpd)', xaxis_title='Dia', yaxis_title='Barris por Dia')
fig.show()
