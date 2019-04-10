import numpy as np
from functions import ReLU, Softmax
from logger import Logger
import optimize

class Network:

    def __init__(self, hidden_shapes, input_shape=784, output_shape=10, weight_scale=1e-3, xavier=False):
        """Network designet to work on MNIST dataset.
        
        Arguments:
            hidden_shapes -- list of shapes of each layer
        
        Keyword Arguments:
            input_shape --  (default: {784})
            output_shape -- (default: {10})
            weight_scale -- Weights will be multiplied by this number during initialization(default: {1e-3})
            xavier -- Xavier initialization (default: {False})
        """

        shapes = [input_shape] + hidden_shapes + [output_shape]             # Shapes of all layers
        self.shapes = shapes                                                
        self.L = len(hidden_shapes) + 1                                     # Length of the network
        self.params = self.initialize_params(shapes, weight_scale, xavier)  # Weights and biases (ang gammas, betas)
        self.logger = Logger()
        self.use_batchnorm = False
        self.use_dropout = False
        self.bn_params = None
        
    def initialize_method(self, optim_parameters):
        """Pick gradient descent method"""

        step = None 
        method = optim_parameters.get('method', 'sgd')                      # method of optimalization

        if str.lower(method) == 'sgd':
            step = optimize.SGD_step

        elif str.lower(method) == 'adam':
            step = optimize.Adam_step
            beta1 = optim_parameters.get('beta1', 0.9)
            beta2 = optim_parameters.get('beta2', 0.99)
            s = {key: 0 for key in self.params.keys()}
            v = {key: 0 for key in self.params.keys()}
            optim_parameters['s'] = s
            optim_parameters['v'] = v
            optim_parameters['beta1'] = beta1
            optim_parameters['beta2'] = beta2 
            
        else:
            print("Wrong optimalization method")
        
        return step

    def initialize_params(self, shapes, weight_scale, xavier):
        """Initialize weights and biases"""
        
        params = {}
        for l in range(1, len(shapes)):
            W = np.random.randn(shapes[l - 1], shapes[l]) 
            b = np.zeros((1, shapes[l]))

            if xavier:
                W *= np.sqrt(2/shapes[l - 1])
            else:
                W *= weight_scale

            params['W' + str(l)] = W
            params['b' + str(l)] = b
        
        return params

    def initialize_batchnorm(self, shapes, params):
        bn_params = []
        for l in range(1, self.L):
            params['gamma' + str(l)] = np.ones(shapes[l])
            params['beta' + str(l)] = np.zeros(shapes[l])
            bn_params.append({})
            
        return params, bn_params

    def forward_linear(self, A, W, b):
        Z = A.dot(W) + b
        cache = (A, W, b)
        return Z, cache

    def forward_activation(self, Z):
        A = ReLU.apply(Z)
        cache = Z
        return A, cache

    def backward_linear(self, dZ, linear_cache):
        A_prev, W, b = linear_cache
        m = A_prev.shape[0]

        dW = A_prev.T.dot(dZ) 
        db = np.sum(dZ, axis=0, keepdims=True)
        dA_prev = dZ.dot(W.T)

        return dW, db, dA_prev

    def backward_activation(self, dA, relu_cache):
        Z = relu_cache
        dZ = ReLU.derivative(Z) * dA

        return dZ

    def forward_pass(self, X, params, bn_params, dropout=1, use_batchnorm=False, mode='train'):
        """Compute forward pass. Layers: (Linear -> Relu) * L - 1 -> Linear -> Softmax
        
        Arguments:
            X -- Input matrix
            params -- Parameters dict (Weights and biases
        
        Returns:
            scores -- Matrix of scores of each class (It can be interpreted as probabilities)
            caches -- List of caches for each layer 
        """

        L = self.L # number of layers
        caches = []

        A_prev = X
        for l in range(1, L + 1):
            W = params['W' + str(l)]
            b = params['b' + str(l)]

            Z, linear_cache = self.forward_linear(A_prev, W, b)
            relu_cache, batchnorm_cache, dropout_cache = None, None, None

            # Skip the activation function
            if l == L:
                A = Z
            else:

                A, relu_cache = self.forward_activation(Z)

                if use_batchnorm:
                    bn_param = bn_params[l-1]
                    A, batchnorm_cache = self.batchnorm_forward(A, params['beta' + str(l)], 
                                                                params['gamma' + str(l)], bn_param, mode)

                if self.use_dropout:
                    A, dropout_cache = self.dropout_forward(A, dropout)
            
            caches.append((linear_cache, relu_cache, batchnorm_cache, dropout_cache))
            A_prev = A


        return A, caches

    def backward_pass(self, scores, Y, params, caches, use_batchnorm=False, reg=0):
        """Perform backpropagation. Return gradiets of weights and biases.
        
        Arguments:
            scores -- Result of forward propagation
            Y -- Ground truth labels
            params -- Dict of weights and biases
            caches -- List od caches (from forward propagation)
        
        Returns:
            grads -- Dict of gradients for each layer
            loss -- Loss value
        """

        L = self.L
        # compute loss and derivative of cost function
        loss = Softmax.apply(scores, Y)

        dscores = Softmax.derivative(scores, Y)

        grads = {}

        dA = None
        # Propagate backwards through layers L, L-1, ..., 1
        for l in reversed(range(1, L+1)):

            linear_cache, relu_cache, batchnorm_cache, dropout_cache = caches[l - 1]

            # If it's a last layer, perform linear backprop
            if l == L:
                dW, db, dA = self.backward_linear(dscores, linear_cache)

            # Do activation -> linear backprop
            else:
                if self.use_dropout:
                    dA = self.dropout_backward(dA, dropout_cache)
                
                if use_batchnorm:
                    dZ, dbeta, dgamma = self.batchnorm_backward(dA, batchnorm_cache)
                    grads['beta' + str(l)] = dbeta
                    grads['gamma' + str(l)] = dgamma

                dZ = self.backward_activation(dA, relu_cache)
                dW, db, dA = self.backward_linear(dZ, linear_cache)

            loss += 0.5 * reg * np.sum(np.sum(params['W' + str(l)]**2))
            dW += reg *  params['W' + str(l)]

            grads['W' + str(l)] = dW
            grads['b' + str(l)] = db


        return grads, loss
    
    def dropout_forward(self, A, p):
        """Dropout layer should be placed after activation function. Neurons in activation layer will be
        kept with probability of p
        
        Arguments:
            A  -- activations
            p -- probability of keeping neuron
        """

        mask = np.ones_like(A)
        probs = np.random.rand(A.shape[0], A.shape[1])
        mask[probs > p] = 0
        A *= mask/p

        return A, mask

    def dropout_backward(self, dA, dropout_cache):
        dA *= dropout_cache
        return dA

    def forward_cost(self, X, Y, params):
        """for gradient check"""
        scores, _ = self.forward_pass(X, params, self.bn_params, mode='test')
        loss = Softmax.apply(scores, Y)
        return loss

    def batchnorm_forward(self, x, beta, gamma, bn_param, mode='train', eps=1e-7):
        """Batch-normalization layer. If mode is set to 'train' it computes mean on mini-batch of data
        and uses it to normalize activations. When mode is set to 'test' it uses the running avarages to
        approximate mean and variance"""

        N, D = x.shape
        running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
        running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
        momentum = bn_param.get('momentum', 0.9)
        cache = None

        if mode == 'train':

            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            inv_var = 1 / np.sqrt(var + eps)
            x_hat = (x - mean) * inv_var

            out = x_hat * gamma + beta

            # Save running averages for a test pass
            bn_param['running_mean'] = momentum * running_mean + (1.0 - momentum) * mean
            bn_param['running_var'] = momentum * running_var + (1.0 - momentum) * var
            cache = (inv_var, x_hat, gamma) 

        else:
            # Get running averages and use them to normalize activations
            mean = running_mean
            var = running_var

            inv_var = 1 / np.sqrt(var + eps)
            x_hat = (x - mean) * inv_var
            out = x_hat * gamma + beta
        
        return out, cache

    def batchnorm_backward(self, dout, cache):
        inv_var, x_hat, gamma = cache
        N, D = dout.shape
        dx_hat = dout * gamma

        dgamma = np.sum(x_hat * dout, axis=0)
        dbeta = np.sum(dout, axis=0)
        dx = (1 / N) * inv_var * (N * dx_hat 
                                 - np.sum(dx_hat, axis=0) 
                                 - x_hat * np.sum(dx_hat * x_hat, axis=0)) 

        return dx, dbeta, dgamma

    def train(self, X_train, Y_train, batch_size, epochs, optim_parameters={},
              verbose=False, X_test=None, Y_test=None):
        """Optimize loss function. 
        
        Arguments:
            X_train -- Matrix of inputs
            Y_train -- Vector of labels
            batch_size -- size of mini-batch
            epochs -- number of epochs (walks through dataset)
        
        Keyword Arguments:
            optim_parameters -- {'method': 'sgd' or 'adam',
                                 'learning_rate": real number,
                                 'use_bachnorm': boolean,
                                 'dropout': number between [0, 1], propability of keeping a neuron,
                                 // adam parameters
                                 'beta1', 'beta2', number between [0, 1]} (default: {{}})
            X_test -- test data
            Y_test -- test labels
            verbose -- verbose (default: {False})
        """

        reg = optim_parameters.get('reg', 0)
        dropout = optim_parameters.get('dropout', 1)
        lr_decay = optim_parameters.get('lr_decay', 1)
        use_batchnorm = optim_parameters.get('use_batchnorm', False)

        # Set flags
        self.use_batchnorm = use_batchnorm
        self.use_dropout = False if dropout == 1 else True

        # Init batchnorm parameters if necessary
        if use_batchnorm:
            self.params, self.bn_params = self.initialize_batchnorm(self.shapes, self.params)
        else:
            self.bn_params = None

        # set method of gradient descent
        step = self.initialize_method(optim_parameters)

        # set batchnorm mode to train
        mode = 'train'

        it = 0
        for epoch in range(epochs):

            # Shuffle X, Y and make mini batches
            X_mini_batches, Y_mini_batches = self.batchify(X_train, Y_train, batch_size)

            for x_mb, y_mb in zip(X_mini_batches, Y_mini_batches):
                scores, caches = self.forward_pass(x_mb, self.params, self.bn_params, dropout=dropout, mode=mode, use_batchnorm=use_batchnorm)
                grads, loss = self.backward_pass(scores, y_mb, self.params, caches, reg=reg, use_batchnorm=use_batchnorm)

                # log loss
                self.logger.add_loss(loss)

                # Perform gradient descent step (method in optim_paramters)
                self.params, optim_parameters = step(self.params, grads, optim_parameters)

                if verbose and it % 100 == 0:
                    print("Epoch {}/{}, loss: {}, lr: {}".format(epoch, epochs, loss, optim_parameters['learning_rate']))
                it += 1

            # log errors
            if X_test is not None and Y_test is not None:
                val_error = 1 - self.accuracy(X_test, Y_test)
                train_error = 1 - self.accuracy(X_train, Y_train)
                self.logger.add_errors(val_error, train_error)

            # decay learing rate
            optim_parameters['learning_rate'] *= lr_decay

    def batchify(self, X, Y, batch_size):
        """ Shuffle and make mini batches"""

        indexes = np.arange(X.shape[0])
        np.random.shuffle(indexes)
        
        X_shuffeled = X[indexes]
        Y_shuffeled = Y[indexes]
        
        X_mini_batches = []
        Y_mini_batches = []

        for k in np.arange(0, X.shape[0], batch_size):
            X_mini_batches.append(X_shuffeled[k : k + batch_size, :])
            Y_mini_batches.append(Y_shuffeled[k : k + batch_size])

        return X_mini_batches, Y_mini_batches

    def predict(self, X):
        """Predict Y labels
        
        Arguments:
            X  -- Matrix of inputs
        
        Returns:
            vector -- Predicted labels
        """

        scores, _ = self.forward_pass(X, self.params, self.bn_params,
                                     use_batchnorm=self.use_batchnorm, mode='test')
        y_predict = np.argmax(scores, axis=1)
        return y_predict
    
    def accuracy(self, X, Y_true):
        y_predict = self.predict(X)
        return np.mean(y_predict == Y_true)

    def get_logger(self):
        return self.logger
