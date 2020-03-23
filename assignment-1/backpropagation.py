import math
import numpy as np
x = 3
y = -4

# forward pass
sigy = 1.0 / (1 + math.exp(-y))
num = x + sigy
sigx = 1.0 / (1 + math.exp(-x))
xpy = x + y
xpysqr = (x+y)**2
den = sigx + xpysqr
invden = 1.0 / den
f = num * invden
print(f)

# backprop f = bum * invden
dnum = invden
dinvden = num

# backprop invden = 1.0 / den
dden = (-1.0 / (den**2)) * dinvden

# backprop den = sigx + xpysqr
dsigx = 1 * dden
dxpysqr = 1 * dden

# backprop xpyspr = xpy ** 2
dxpy = (2 * xpy) * dxpysqr

# backprop xpy = x + y
dy = 1 * dxpy
dx = 1 * dxpy

# backprop sigx = 1.0 / (1 + math.exp(-x))
dx += ((1 - sigx) * sigx) * dsigx

# backprop num = x + sigy
dx = 1 * dnum
dsigy = 1 * dnum

# backprop sigy = 1.0 / (1 + math.exp(-y))
dy += ((1 - sigy) * sigy) * dsigy

# done
# Gradients for vectorized operations

W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)

dD = np.random.randn(*D.shape)
dW = dD.dot(X.T)
dX = W.T.dot(dD)
print(dD)
















