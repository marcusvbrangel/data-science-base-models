
import numpy as np

#----------------------------------------------------------
# algebras linear - multiplicacao...
#----------------------------------------------------------

# duas matrizes/vetores do mesmo tamanho...
a = np.array([2, 3, 4])
b = np.array([5, 6, 7])

# 1- multiplicacao elemento a elemento (Hadamard)...
resultado = a * b
print(resultado)   # [10 18 28] ...


# 2- multiplicacao de matrizes (Matrix Multiplication)...

# matriz 2 linhas por 3 colunas...
c = np.array([[1, 2, 3],
              [4, 5, 6]])

d = np.array([[7, 8],
              [9, 10],
              [11, 12]])

resultado = c @ d     # ou np.matmul(c, d) ...
print(resultado)



# 3. multiplicacao na unha...
'''
Regra da Multiplicação
Para multiplicar matrizes: cada linha de c multiplica cada coluna de d.
Resultado será uma matriz 2x2 (2 linhas de c × 2 colunas de d)
'''

# Posição [0, 0] (linha 0 de c × coluna 0 de d)...
posicao_0_0 = (1 * 7) + (2 * 9) + (3 * 11)
print(posicao_0_0)

# Posição [0, 1] (linha 0 de c × coluna 1 de d)...
posicao_0_1 = (1 * 8) + (2 * 10) + (3 * 12)
print(posicao_0_1)

# Posição [1, 0] (linha 1 de c × coluna 0 de d)...
posicao_1_0 = (4 * 7) + (5 * 9) + (6 * 11)
print(posicao_1_0)

# Posição [1, 1] (linha 1 de c × coluna 1 de d)...
posicao_1_1 = (4 * 8) + (5 * 10) + (6 * 12)
print(posicao_1_1)

resultado_final = [[posicao_0_0, posicao_0_1],
                   [posicao_1_0, posicao_1_1]]

print(resultado_final)


# ------------------------------------------------------------
# casos de uso praticos/reais...
# ------------------------------------------------------------

# 1- desconto em produtos...

# precos originais de 3 produtos...
precos = np.array([100, 50, 200])

# desconto aplicado a cada produto (0.8 = 20% de desconto) ...
desconto = np.array([0.8, 0.9, 0.7])

# precos finais (multiplicacao elemento a elemento) ...
precos_finais = precos * desconto
print(f"precos_finais: {precos_finais}")


# 2- Vendas por Produto...

# quantidade vendida de cada produto em 3 lojas (3 x 3) ...
# cada linha e uma loja...
# cada coluna e um produto...
quantidade_vendida = np.array([[10, 5, 8],
                               [15, 12, 9],
                               [7, 6, 4]])

# precos de cada produto...
preco_produto = np.array([50, 100, 75])

# receita por loja (multiplicacao de matrizes) ...
receita = quantidade_vendida @ preco_produto
print(f"receita: {receita}")

# Loja 1: (10×50) + (5×100) + (8×75) = 500 + 500 + 600 = 1600
# Loja 2: (15×50) + (12×100) + (9×75) = 750 + 1200 + 675 = 2625
# Loja 3: (7×50) + (6×100) + (4×75) = 350 + 600 + 300 = 1250


'''
Diferença Resumida

Tipo	                Código	            Resultado	    Caso de Uso
-----------------------------------------------------------------------------------------------
Elemento-a-Elemento	    a * b	            Vetor/Matriz	Descontos, ajustes por elemento
Matrizes	            A @ B	            Matriz	        Transformações, cálculos combinados
Produto Escalar	        a @ b ou a.dot(b)	Escalar	        Machine learning, predições

A escolha depende do que você quer fazer com seus dados!
'''

