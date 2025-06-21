import numpy as np
import pandas as pd

class MedicalRNN:
    """
    RNN for medical diagnosis with binary classification.
    Adapted for structured data rather than sequential data.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the RNN for medical diagnosis.

        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layer
            output_size: Size of output (1 for binary classification)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights with Xavier/Glorot initialization
        self.W_input_hidden = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.W_hidden_hidden = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (hidden_size + hidden_size))
        self.W_hidden_output = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / (hidden_size + output_size))

        # Initialize biases
        self.hidden_bias = np.zeros((hidden_size, 1))
        self.output_bias = np.zeros((output_size, 1))

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x):
        """
        Forward pass for a single sample.

        Args:
            x: Input features (input_size, 1)
        """
        # Reshape input if necessary
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        # Initialize hidden state
        hidden = np.zeros((self.hidden_size, 1))

        # Compute hidden state
        hidden = np.tanh(np.dot(self.W_input_hidden, x) +
                         np.dot(self.W_hidden_hidden, hidden) +
                         self.hidden_bias)

        # Compute output (sigmoid for binary classification)
        output = self.sigmoid(np.dot(self.W_hidden_output, hidden) +
                              self.output_bias)

        return output, hidden

    def compute_loss(self, y_pred, y_true):
        """Binary cross-entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def train(self, X_train, y_train, epochs=10, batch_size=32, learning_rate=0.01):
        """
        Train the model on the medical dataset.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples, 1)
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            learning_rate: Learning rate for gradient descent
        """
        n_samples = X_train.shape[0]
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            # Create mini-batches
            indices = np.random.permutation(n_samples)

            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                batch_loss = 0
                # Initialize gradients
                dW_input_hidden = np.zeros_like(self.W_input_hidden)
                dW_hidden_hidden = np.zeros_like(self.W_hidden_hidden)
                dW_hidden_output = np.zeros_like(self.W_hidden_output)
                db_hidden = np.zeros_like(self.hidden_bias)
                db_output = np.zeros_like(self.output_bias)

                # Process each sample in batch
                for i in range(len(X_batch)):
                    x = X_batch[i].reshape(-1, 1)
                    y = y_batch[i].reshape(-1, 1)

                    # Forward pass
                    output, hidden = self.forward(x)

                    # Compute loss
                    batch_loss += self.compute_loss(output, y)

                    # Backward pass
                    # Output layer gradients
                    d_output = output - y
                    dW_hidden_output += np.dot(d_output, hidden.T)
                    db_output += d_output

                    # Hidden layer gradients
                    d_hidden = np.dot(self.W_hidden_output.T, d_output) * (1 - hidden * hidden)
                    dW_input_hidden += np.dot(d_hidden, x.T)
                    dW_hidden_hidden += np.dot(d_hidden, hidden.T)
                    db_hidden += d_hidden

                # Update weights with average gradients
                batch_size_actual = len(X_batch)
                self.W_input_hidden -= learning_rate * dW_input_hidden / batch_size_actual
                self.W_hidden_hidden -= learning_rate * dW_hidden_hidden / batch_size_actual
                self.W_hidden_output -= learning_rate * dW_hidden_output / batch_size_actual
                self.hidden_bias -= learning_rate * db_hidden / batch_size_actual
                self.output_bias -= learning_rate * db_output / batch_size_actual

                epoch_loss += batch_loss / batch_size_actual

            avg_epoch_loss = epoch_loss / (n_samples / batch_size)
            losses.append(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

        return losses

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Predicted probabilities (n_samples, 1)
        """
        predictions = []
        for i in range(len(X)):
            output, _ = self.forward(X[i])
            predictions.append(output)
        return np.array(predictions)

    def evaluate(self, X, y_true):
        """
        Evaluate model performance.

        Args:
            X: Input features
            y_true: True labels

        Returns:
            Dictionary with accuracy, precision, recall, and F1 score
        """
        y_pred = self.predict(X)
        y_pred_binary = (y_pred >= 0.5).astype(int)

        accuracy = np.mean(y_pred_binary == y_true)

        # Calculate other metrics
        tp = np.sum((y_pred_binary == 1) & (y_true == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

