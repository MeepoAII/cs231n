# training a Neural Network
import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2
K = 3
X = np.zeros((N*K, D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8')
reg = 1e-3
step_size = 1e-0

for i in range(K):
    ix = range(N*i, N*(i+1))
    r = np.linspace(0.0, 1, N)
    t = np.linspace(i*4, (i+1)*4, N) + np.random.rand(N)*0.2
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = i

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()


h = 100
W = 0.01 * np.random.randn(D, h)
b = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))

num_example = X.shape[0]
for i in range(10000):

    # evaluate class scores, [NxK]
    hidden_layer = np.maximum(0, np.dot(X, W) + b)
    scores = np.dot(hidden_layer, W2) + b2

    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(num_example), y])
    data_loss = np.sum(correct_logprobs) / num_example
    reg_loss = 0.5 * reg * np.sum(W*W) + 0.5 * reg * np.sum(W2*W2)
    loss = data_loss + reg_loss

    if i % 1000 == 0:
        print("itereation %d: loss %f" % (i, loss))

    # compute the gradient on scores
    dscores = probs
    dscores[range(num_example), y] -= 1
    dscores /= num_example

    # backpropate the gradient to the parameters
    # first backprop into parameters W2 and b2
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)

    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)

    # backprop the ReLU no-linearity
    dhidden[hidden_layer <= 0] = 0

    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)

    # add regularization gradient contribution
    dW2 += reg * W2
    dW += reg * W

    # perform a parameter update
    W += -step_size * dW
    b += -step_size *db
    W2 += -step_size * dW2
    b2 += -step_size * db2

# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print("training accuracy: %.2f" % (np.mean(predicted_class == y)))







