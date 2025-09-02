#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt


def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    return np.tanh(x)

def relu_activation(x):
    return np.maximum(0, x)


class Perceptron:
    def __init__(self, activation_function):
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)
        self.activation_function = activation_function
        
    def predict(self, inputs):
        total_input = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(total_input)
    
    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            for input_vector, target in zip(inputs, targets):
                prediction = self.predict(input_vector)
                error = target - prediction
                self.weights += learning_rate * error * input_vector
                self.bias += learning_rate * error

            if epoch % 100 == 0:
                loss = np.mean(np.square(targets - self.predict(inputs)))
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")


inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])  

targets = np.array([0, 0, 0, 1])

activation_functions = {
    "Sigmoid": sigmoid_activation,
    "Tanh": tanh_activation,
    "ReLU": relu_activation
}

for name, activation_function in activation_functions.items():
    print(f"\nTraining Perceptron with {name} Activation:")
    perceptron = Perceptron(activation_function)
    perceptron.train(inputs, targets, epochs=1000, learning_rate=0.1)
    predictions = [perceptron.predict(x) for x in inputs]
    print(f"Predictions using {name} activation: {predictions}")


def plot_activation_functions():
    x = np.linspace(-10, 10, 400)
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.plot(x, sigmoid_activation(x), label='Sigmoid')
    plt.title('Sigmoid Activation')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(x, tanh_activation(x), label='Tanh')
    plt.title('Tanh Activation')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(x, relu_activation(x), label='ReLU')
    plt.title('ReLU Activation')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


plot_activation_functions()


# In[ ]:


0

