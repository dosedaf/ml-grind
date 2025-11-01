import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def init_param(input_size, hidden1_size, hidden2_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden1_size) * 0.01
    b1 = np.zeros((1, hidden1_size))
    W2 = np.random.randn(hidden1_size, hidden2_size) * 0.01
    b2 = np.zeros((1, hidden1_size))
    W3 = np.random.randn(hidden2_size, output_size) * 0.01
    b3 = np.zeros((1, output_size))
    
    return W1, b1, W2, b2, W3, b3

def forward_pass(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(X, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(X, W3) + b3
    A3 = relu(Z3)

def backward_pass(X, Y, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3):
    m = X.shape([0])

    dZ3 = A3 - Y
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dZ2 = np.dot(dZ3, W3.T) * relu_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dZ1 = np.dot(dZ2, W2.T) * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
