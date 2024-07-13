
import numpy as np


class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.transfer_function = transfer_function
        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)
        self.weights = np.random.randn(self.number_of_nodes, self.input_dimensions)
    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        if np.shape(W) != np.shape(self.weights):
            return -1
        self.weights = W
    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """
        return self.weights
    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        ''' multiplying the weights and inputs for net '''
        self.net_matrix = np.dot(self.weights, X)

        '''applying hardlimit on the net'''
        if self.transfer_function == 'Hard_limit':
          self.net_matrix[self.net_matrix >= 0] = 1
          self.net_matrix[self.net_matrix < 0] = 0

        return (self.net_matrix)
    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        '''finding the input transpose'''
        self.X_tran = X.T
        ''' multiply input matrix transpose with it self to convert into in square matrix'''
        self.Square_matrix = np.dot(self.X_tran, X)
        '''applying inverse on square matrix'''
        self.inverse_matrix = np.linalg.pinv(self.Square_matrix)
        ''' multiplying inverse matrix with the input transpose to get pseudoinverse'''
        self.pseudo_input_matrix = np.dot(self.inverse_matrix, self.X_tran)
        '''adjusting the weights through suedo inverse'''
        self.weights = np.dot(y,self.pseudo_input_matrix)

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        for epoch in range(num_epochs):
            self.X_T = X.T
            self.Y_T = y.T                   
            batches = -(-len(self.X_T)//batch_size)
            b_low = 0
            b_up = batch_size 
            for i in range(batches):
                self.actual = self.predict(X).T
                input_batch = self.X_T[b_low : b_up]
                Target_batch = self.Y_T[b_low : b_up]
                Actual_batch = self.actual[b_low : b_up]
                error = Target_batch - Actual_batch
                
                if learning == 'Filtered':
                  self.weights = np.add(((1-gamma) * self.weights),  (alpha * (np.dot(Target_batch.T,input_batch))))
                elif learning == 'Delta':
                   self.weights = np.add(self.weights,  (np.dot(error.T,input_batch).dot(alpha)))
                elif learning == 'Unsupervised_hebb':
                  self.weights = np.add(self.weights, alpha * (np.dot(Actual_batch.T,input_batch))) 
                b_low = b_low + batch_size
                b_up = b_up + batch_size 
                
    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        self.A = self.predict(X)
        mse = (np.square(y - self.A)).mean(axis=None)
        return mse        