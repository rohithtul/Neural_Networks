
# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """

        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.all_activation_functions = []

    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
         "Linear", "Relu","Sigmoid".
         :return: None
         """    
        self.weights.append(np.random.randn(self.input_dimension, num_nodes))
        self.input_dimension=num_nodes
        self.biases.append(np.random.randn(num_nodes))
        self.all_activation_functions.append(transfer_function)
    
    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
        """
        return self.weights[layer_number]
    
    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        return self.biases[layer_number]
        
    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.weights[layer_number]=weights
        
    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.biases[layer_number]=biases
        
    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        sparse_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
        return tf.reduce_mean(sparse_ce)
        
    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        input_for_layer=X
        for (weights, bias, transferfunction) in zip(self.weights, self.biases, self.all_activation_functions):
            #calculate the net for each layer wx+b
            net=tf.matmul(input_for_layer,weights)
            net=tf.add(net,bias)
            if transferfunction == "Sigmoid":
                input_for_layer = tf.nn.sigmoid(net) 
            elif transferfunction == "Linear":
                input_for_layer = net
            elif transferfunction == "Relu":
                input_for_layer = tf.nn.relu(net)
        #final layer output         
        return input_for_layer
    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
         """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
         batches = tf.data.Dataset.from_tensor_slices((X_train, y_train))
         batches = batches.batch(batch_size)
         for epoch in range(num_epochs):
            for batch_num, (c,d) in enumerate(batches):
                with tf.GradientTape() as tape:
                    actual = self.predict(c)
                    batch_loss = self.calculate_loss(d, actual)
                    dloss_dw, dloss_db = tape.gradient(batch_loss, [self.weights, self.biases])
                    for s in range(len(self.weights)):
                        self.weights[s].assign_sub(alpha * dloss_dw[s])
                        self.biases[s].assign_sub(alpha * dloss_db[s])
         
    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        error = 0
        target_op = y
        actual_op = self.predict(X)
        actual_op = np.argmax(actual_op, axis =1)
        for p in range(len(actual_op)):
            if target_op[p] != actual_op[p]:
                error = error + 1                
        percent_error = error/len(actual_op)
        return percent_error
       
    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        actual_op = self.predict(X)
        actual_op=np.argmax(actual_op,axis=1)
        return tf.math.confusion_matrix(y,actual_op)