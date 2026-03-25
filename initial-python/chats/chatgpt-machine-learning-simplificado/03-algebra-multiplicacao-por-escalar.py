
import numpy as np

#----------------------------------------------------------
# algebras linear - multiplicacao por escalar...
#----------------------------------------------------------

'''
Como Funciona
Quando você multiplica uma matriz por um número, cada elemento é 
multiplicado por esse número...

Multiplicação por escalar é fundamental em machine learning, 
processamento de imagens, transformações de dados e cálculos científicos!...
'''

# multiplicando um vetor...
a = np.array([1, 2, 3])
resultado = a * 10
print(resultado)  # [10 20 30]


# multiplicando uma matriz...
d = np.array([[7, 8],
              [9, 10],
              [11, 12]])

resultado = d * 10
print(resultado)
# [[70, 80],
#  [90, 100],
#  [110, 120]]


# ----------------------------------------------------
# casos de uso praticos/reais...
# ----------------------------------------------------


# 1- ajuste de escala (zoom)...

# imagem original (valores em pixels)...
imagem = np.array([[100, 150],
                  [200, 50]])

# aumentar brilho multiplicando por 1.5 ...
imagem_brilho = imagem * 1.5
print(f"imagem_brilho: {imagem_brilho}")


# 2- conversao de unidades...

# precos em real...
precos_real = np.array([100, 200, 300])

# converter preco para centavos...
precos_centavos = precos_real * 100
print(f"precos_centavos: {precos_centavos}")


# 3- ajuste de taxa/percentual...

# valores de vendas...
vendas = np.array([[1000, 2000],
                   [1500, 800]])

# aplicar crestimento de 20% (multiplicar por 1.2) ...
vendas_crescimento = vendas * 1.2
print(f"vendas_crescimento: {vendas_crescimento}")


# pesos de uma rede neural...
pesos = np.array([0.5, 0.3, 0.8])

# aumentar importancia dos pesos (baseado na taxa de aprendizado) ...
pesos_ajustados = pesos * 0.01   # taxa de aprendizado...
print(f"pesos_ajustados: {pesos_ajustados}")


'''
Propriedades Matemáticas
Comutativa: matriz * 10 = 10 * matriz
Distributiva: (A + B) * 10 = A*10 + B*10
Associativa: (matriz * 5) * 2 = matriz * (5 * 2) = matriz * 10
'''
