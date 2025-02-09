import numpy as np
import pandas as pd


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output

        # Bias terms
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h_t = np.zeros((self.hidden_size, 1))  # Initial hidden state
        outputs, hidden_states = [], []

        inputs = inputs.reshape(-1, self.input_size, 1)  # Ensure shape (time steps, features, 1)
        for t in range(inputs.shape[0]):  # Iterate over time steps
            x_t = inputs[t]  # No need to reshape again

            h_t = np.tanh(np.dot(self.Wxh, x_t) + np.dot(self.Whh, h_t) + self.bh)
            y_t = 1 / (1 + np.exp(-np.dot(self.Why, h_t) - self.by))  # Sigmoid activation
            outputs.append(y_t)
            hidden_states.append(h_t)

        return outputs, hidden_states

    def backward(self, inputs, targets, outputs, hidden_states):
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(inputs))):
            x_t = inputs[t].reshape(-1, 1)
            h_t = hidden_states[t]
            y_t = outputs[t]
            dy = y_t - targets[t]  # Binary cross-entropy derivative

            dWhy += np.dot(dy, h_t.T)
            dby += dy

            dh = np.dot(self.Why.T, dy) + dh_next
            dh_raw = (1 - h_t ** 2) * dh  # tanh derivative

            dWxh += np.dot(dh_raw, x_t.T)
            dWhh += np.dot(dh_raw, hidden_states[t - 1].T if t > 0 else np.zeros_like(h_t).T)
            dbh += dh_raw

            dh_next = np.dot(self.Whh.T, dh_raw)

        # Update weights
        for param, dparam in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                 [dWxh, dWhh, dWhy, dbh, dby]):
            param -= self.learning_rate * dparam

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            loss = 0
            for i in range(len(X)):
                inputs = X[i]
                target = y[i]
                outputs, hidden_states = self.forward(inputs)
                loss += np.mean(-(target * np.log(outputs[-1]) + (1 - target) * np.log(1 - outputs[-1])))
                self.backward(inputs, target, outputs, hidden_states)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss / len(X)}")

