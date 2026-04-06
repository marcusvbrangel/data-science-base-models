import numpy as np
import matplotlib.pyplot as plt

tempo = np.arange(10)
p1 = [10, 12, 11, 10, 9, 8, 7, 6, 5, 4]
p2 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

plt.stackplot(tempo, p1, p2, labels=['Campo Velho', 'Poço Novo'], colors=['#ff9999','#66b3ff'])
plt.legend(loc='upper left')
plt.title("Composição da Produção da FPSO")
plt.show()
