
import numpy as np

#----------------------------------------------------------
# algebra linear - produto escalar...
#----------------------------------------------------------


x = np.array([10, 2000])   # features...
w = np.array([0.5, 0.1])   # pesos...

# usando biblioteca numpy...
y = x.dot(w)
print(y)

# ou na unha mesmo...
z = x[0] * w[0] + x[1] * w[1]
print(z)

# ou operador nativo/novo do python '@'...
k = x @ w
print(k)
