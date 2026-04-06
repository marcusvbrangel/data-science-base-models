import plotly.express as px
import numpy as np

# Simulando trajetória: Profundidade (Z), Leste (X), Norte (Y)
z = np.linspace(0, 5000, 100)
x = 500 * np.sin(z/1000)
y = 300 * np.cos(z/1000)

fig = px.line_3d(x=x, y=y, z=-z, title="Trajetória 3D do Poço Offshore")
fig.show()
