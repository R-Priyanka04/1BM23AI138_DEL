#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, activation='linear'):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.activation = activation

    def activate(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'softmax':
            
            return x  
        else:
            return x

    def softmax_batch(self, x_array):
        exps = np.exp(x_array - np.max(x_array))
        return exps / np.sum(exps)

    def predict(self, inputs):
        linear_output = np.dot(inputs, self.weights) + self.bias
        if self.activation == 'softmax':
            
            return linear_output
        return self.activate(linear_output)

    def train(self, training_inputs, labels, epochs=100, lr=0.1):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                output = self.predict(inputs)
                error = label - output
                self.weights += lr * error * inputs
                self.bias += lr * error


inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
labels = np.array([0, 0, 0, 1])


p_linear = Perceptron(2, activation='linear')
p_linear.train(inputs, labels)
out_linear = [p_linear.predict(i) for i in inputs]

p_sigmoid = Perceptron(2, activation='sigmoid')
p_sigmoid.train(inputs, labels)
out_sigmoid = [p_sigmoid.predict(i) for i in inputs]

p_tanh = Perceptron(2, activation='tanh')
p_tanh.train(inputs, labels)
out_tanh = [p_tanh.predict(i) for i in inputs]

p_relu = Perceptron(2, activation='relu')
p_relu.train(inputs, labels)
out_relu = [p_relu.predict(i) for i in inputs]

p_softmax = Perceptron(2, activation='softmax')
p_softmax.train(inputs, labels)
logits_softmax = [p_softmax.predict(i) for i in inputs]
out_softmax = p_softmax.softmax_batch(np.array(logits_softmax))

print("Predictions using Linear activation:")
for x, y in zip(inputs, out_linear):
    print(f"Input: {x}, Output: {y:.4f}")

print("\nPredictions using Sigmoid activation:")
for x, y in zip(inputs, out_sigmoid):
    print(f"Input: {x}, Output: {y:.4f}")

print("\nPredictions using Tanh activation:")
for x, y in zip(inputs, out_tanh):
    print(f"Input: {x}, Output: {y:.4f}")

print("\nPredictions using ReLU activation:")
for x, y in zip(inputs, out_relu):
    print(f"Input: {x}, Output: {y:.4f}")

print("\nPredictions using Softmax activation:")
for x, y in zip(inputs, out_softmax):
    print(f"Input: {x}, Output: {y:.4f}")

x_axis = ['[0,0]', '[0,1]', '[1,0]', '[1,1]']
plt.plot(x_axis, out_linear, label='Linear', marker='o')
plt.plot(x_axis, out_sigmoid, label='Sigmoid', marker='s')
plt.plot(x_axis, out_tanh, label='Tanh', marker='^')
plt.plot(x_axis, out_relu, label='ReLU', marker='x')
plt.plot(x_axis, out_softmax, label='Softmax', marker='d')
plt.ylabel('Output')
plt.title('Perceptron outputs with different activation functions')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




