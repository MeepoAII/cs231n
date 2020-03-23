import numpy as np
import matplotlib.pyplot as plt

# get some data
N = 100
D = 2
K = 3
X = np.zeros((N*K, D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8')
reg = 1e-3
step_size = 1e-0

for i in range(K):
    ix = range(N*i, N*(i+1))
    r = np.linspace(0, 0.1, N)
    t = np.linspace(i*4, (i+1)*4, N) + np.random.rand(N)*0.2
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = i

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))
num_examples = X.shape[0]

for i in range(200):
    scores = X.dot(W) + b

    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W*W)
    loss = data_loss + reg_loss

    if i % 10 == 0:
        print(f"iteration {i}: loss {loss}")

    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    # backpropate the gradient to the parameters W, b
    dW = X.T.dot(dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    dW += reg * W

    W += -step_size * dW
    b += -step_size * db

scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print(f"training accuracy: %.2f" % (np.mean(predicted_class == y)))
