import numpy as np


class RecurrentNeuralNetwork:
    """
    A simple Recurrent Neural Network (RNN) implementation.

    This RNN processes sequences of input vectors and produces output predictions
    at each time step. It uses tanh activation for hidden states and softmax for outputs.

    Attributes:
        input_dim (int): Dimension of input vectors
        hidden_dim (int): Dimension of hidden state
        output_dim (int): Dimension of output vectors
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the RNN with given dimensions and random weights.

        Args:
            input_dim: Size of input vectors
            hidden_dim: Size of hidden state
            output_dim: Size of output vectors
        """
        # Store dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights with small random values to break symmetry
        self.W_input_hidden = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W_hidden_hidden = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_hidden_output = np.random.randn(output_dim, hidden_dim) * 0.01

        # Initialize biases with zeros
        self.hidden_bias = np.zeros((hidden_dim, 1))
        self.output_bias = np.zeros((output_dim, 1))

        # Storage for forward pass computations
        self.memory = None

    def forward(self, input_sequence):
        """
        Perform forward propagation through time.

        Args:
            input_sequence: List of input vectors, each shape (input_dim, 1)

        Returns:
            tuple: (probabilities, hidden_states)
                - probabilities: Dict of output probabilities at each time step
                - hidden_states: Dict of hidden states at each time step
        """
        # Initialize storage for this sequence
        self.memory = {
            'inputs': {},  # Store input vectors
            'hidden_states': {},  # Store hidden states
            'outputs': {},  # Store raw outputs
            'probabilities': {}  # Store output probabilities
        }

        # Initialize first hidden state with zeros
        current_hidden_state = np.zeros((self.hidden_dim, 1))
        self.memory['hidden_states'][-1] = current_hidden_state

        # Process each time step
        for t, input_vector in enumerate(input_sequence):
            self.memory['inputs'][t] = input_vector

            # 1. Compute hidden state
            # Combine input and previous hidden state
            hidden_input = (np.dot(self.W_input_hidden, input_vector) +
                            np.dot(self.W_hidden_hidden, current_hidden_state) +
                            self.hidden_bias)
            current_hidden_state = np.tanh(hidden_input)
            self.memory['hidden_states'][t] = current_hidden_state

            # 2. Compute output
            # Generate raw output scores
            output = np.dot(self.W_hidden_output, current_hidden_state) + self.output_bias
            self.memory['outputs'][t] = output

            # 3. Apply softmax for probabilities
            exp_output = np.exp(output)
            probabilities = exp_output / np.sum(exp_output)
            self.memory['probabilities'][t] = probabilities

        return self.memory['probabilities'], self.memory['hidden_states']

    def backward(self, targets, learning_rate=0.01):
        """
        Perform backward propagation through time (BPTT).

        Args:
            targets: List of target indices for each time step
            learning_rate: Step size for gradient descent
        """
        # Initialize gradient accumulators
        gradients = {
            'W_input_hidden': np.zeros_like(self.W_input_hidden),
            'W_hidden_hidden': np.zeros_like(self.W_hidden_hidden),
            'W_hidden_output': np.zeros_like(self.W_hidden_output),
            'hidden_bias': np.zeros_like(self.hidden_bias),
            'output_bias': np.zeros_like(self.output_bias)
        }

        # Initialize gradient of next hidden state
        next_hidden_gradient = np.zeros_like(self.memory['hidden_states'][0])

        # Iterate backwards through time
        for t in reversed(range(len(targets))):
            # 1. Gradient of output layer
            output_gradient = self.memory['probabilities'][t].copy()
            output_gradient[targets[t]] -= 1  # Derivative of cross-entropy loss

            # 2. Update output layer gradients
            gradients['W_hidden_output'] += np.dot(output_gradient,
                                                   self.memory['hidden_states'][t].T)
            gradients['output_bias'] += output_gradient

            # 3. Gradient of hidden layer
            hidden_gradient = (np.dot(self.W_hidden_output.T, output_gradient) +
                               next_hidden_gradient)

            # 4. Gradient through tanh
            # tanh'(x) = 1 - tanh²(x)
            hidden_state = self.memory['hidden_states'][t]
            tanh_gradient = (1 - hidden_state * hidden_state) * hidden_gradient

            # 5. Update hidden layer gradients
            gradients['hidden_bias'] += tanh_gradient
            gradients['W_input_hidden'] += np.dot(tanh_gradient,
                                                  self.memory['inputs'][t].T)
            gradients['W_hidden_hidden'] += np.dot(tanh_gradient,
                                                   self.memory['hidden_states'][t - 1].T)

            # 6. Save gradient for next iteration
            next_hidden_gradient = np.dot(self.W_hidden_hidden.T, tanh_gradient)

        # Clip gradients to prevent explosion
        for grad in gradients.values():
            np.clip(grad, -5, 5, out=grad)

        # Update weights and biases
        self.W_input_hidden -= learning_rate * gradients['W_input_hidden']
        self.W_hidden_hidden -= learning_rate * gradients['W_hidden_hidden']
        self.W_hidden_output -= learning_rate * gradients['W_hidden_output']
        self.hidden_bias -= learning_rate * gradients['hidden_bias']
        self.output_bias -= learning_rate * gradients['output_bias']

    def train_step(self, input_sequence, targets, learning_rate=0.01):
        """
        Perform one training step on a sequence.

        Args:
            input_sequence: List of input vectors
            targets: List of target indices
            learning_rate: Learning rate for parameter updates

        Returns:
            float: Cross-entropy loss for this sequence
        """
        # Forward pass
        probabilities, _ = self.forward(input_sequence)

        # Backward pass
        self.backward(targets, learning_rate)

        # Compute cross-entropy loss
        loss = sum(-np.log(probabilities[t][targets[t]])
                   for t in range(len(targets)))

        return loss