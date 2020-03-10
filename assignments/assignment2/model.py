import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """

        self.reg = reg
        self.hidden_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu_layer = ReLULayer()
        self.output_layer = FullyConnectedLayer(hidden_layer_size, n_output)

        self.n_input = n_input
        self.n_output = n_output
        self.hidden_layer_size = hidden_layer_size

        # TODO Create necessary layers
        # raise Exception("Not implemented!")

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!

        hidden_layer_params = self.hidden_layer.params()
        output_layer_params = self.output_layer.params()

        hidden_layer_params['W'].grad = np.zeros_like(hidden_layer_params['W'].grad)
        hidden_layer_params['B'].grad = np.zeros_like(hidden_layer_params['B'].grad)

        output_layer_params['W'].grad = np.zeros_like(output_layer_params['W'].grad)
        output_layer_params['B'].grad = np.zeros_like(output_layer_params['B'].grad)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        hidden_l_out = self.hidden_layer.forward(X)
        relu_l_out = self.relu_layer.forward(hidden_l_out)
        output_l_out = self.output_layer.forward(relu_l_out)

        ce_loss, d_pred = softmax_with_cross_entropy(output_l_out, y)

        reg_loss_first, d_R_first = l2_regularization(hidden_layer_params['W'].value, self.reg)
        reg_loss_second, d_R_second = l2_regularization(output_layer_params['W'].value, self.reg)

        loss = ce_loss + reg_loss_first + reg_loss_second

        d_input_out_layer = self.output_layer.backward(d_pred)
        output_layer_params['W'].grad += d_R_second

        d_input_relu_layer = self.relu_layer.backward(d_input_out_layer)

        d_input_hidden_layer = self.hidden_layer.backward(d_input_relu_layer)
        hidden_layer_params['W'].grad += d_R_first

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        hidden_l_output = self.hidden_layer.forward(X)
        relu_output = self.relu_layer.forward(hidden_l_output)
        output_l_output = self.output_layer.forward(relu_output)

        pred = np.argmax(output_l_output, axis=1)

        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        hidden_layer_params = self.hidden_layer.params()
        result['W1'] = hidden_layer_params['W']
        result['B1'] = hidden_layer_params['B']

        output_layer_params = self.output_layer.params()
        result['W2'] = output_layer_params['W']
        result['B2'] = output_layer_params['B']

        return result

