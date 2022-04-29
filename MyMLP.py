import numpy as np

class Normalization:
    def __init__(self,):
        self.mean = np.zeros([1,64]) # means of training features
        self.std = np.zeros([1,64]) # standard deviation of training features

    def fit(self,x):
        # compute the statistics of training samples (i.e., means and std)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        pass # placeholder

    def normalize(self,x):
        # normalize the given samples to have zero mean and unit variance (add 1e-15 to std to avoid numeric issue)
        x = (x-self.mean)/(self.std+1e-15)
        return x

def process_label(label):
    # convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label),10])
    for idx, l in enumerate(label):
        one_hot[idx, l] = 1

    return one_hot

def tanh(x):
    # Please use exp(x)-exp(-x)/exp(x)+exp(-x)
    # implement the hyperbolic tangent activation function for hidden layer
    x = np.clip(x,a_min=-100,a_max=100) # for stablility, do not remove this line
    
    f_x = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    return f_x

def softmax(x):
    # Please use exp(xi)/sum_i=1toN exp(xi) to compute
    # implement the softmax activation function for output layer
    f_x = np.exp(x) / sum(np.exp(x))

    return f_x

class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64,num_hid])
        self.bias_1 = np.random.random([1,num_hid])
        self.weight_2 = np.random.random([num_hid,10])
        self.bias_2 = np.random.random([1,10])

    def fit(self, train_x, train_y, valid_x, valid_y):
        """
        This function will train on the training set, and use the validation error to determine when to stop
        Since the lr is given, the final parameters should be consistent to one set of fixed values
        """
        # learning rate
        lr = 5e-3
        # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0

        # normalize the data
        # norm = Normalization()
        # norm.fit(train_x)
        # train_x = norm.normalize(train_x)

        """
        Stop the training if there is no improvment over the best validation accuracy for more than 50 iterations
        """
        
        # print(r.shape)
        while count<=50:
            # training with all samples (full-batch gradient descents)
            # implement the forward pass (from inputs to predictions)

            # starting with data, dot multiply with weights
            alpha = train_x @ self.weight_1 + self.bias_1

            # input into tanh activation function
            z = tanh(alpha)

            # dot multiply
            beta = z @ self.weight_2 + self.bias_2

            # input into softmax activation function
            y = softmax(beta)

            # implement the backward pass (backpropagation)
            # compute the gradients w.r.t. different parameters
            delta_w2 = z.T @ (y - train_y)
            # delta_b2 = np.sum(y - train_y, axis=0) # y - r results in a 1000x4, so we sum over n to get a 1x4
            delta_w1 = train_x.T @ np.multiply((1 - np.power(z,2)), (y - train_y) @ self.weight_2.T)
            # delta_b1 = np.sum(np.multiply((1 - np.power(z,2)), (y - train_y) @ self.weight_2.T), axis=0) # this results in a 1000x10 so we sum to get 1x10

            # update the parameters based on sum of gradients for all training samples
            self.weight_2 = self.weight_2 - lr*delta_w2
            # self.bias_2 = self.bias_2 - lr*delta_b2
            self.weight_1 = self.weight_1 - lr*delta_w1
            # self.bias_1 = self.bias_1 - lr*delta_b1

            # evaluate on validation data
            predictions = self.predict(valid_x) # predictions are length 10 vectors
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)
            # print(valid_acc)
            # compare the current validation accuracy with the best one
            if valid_acc>best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self,x):
        # generate the predicted probability of different classes
        alpha = x @ self.weight_1 + self.bias_1

        # input into tanh activation function
        z = tanh(alpha)

        # dot multiply
        beta = z @ self.weight_2 + self.bias_2

        # input into softmax activation function
        class_probabilities = softmax(beta)

        # convert class probability to predicted labels
        y = np.zeros([len(x),]).astype('int') # placeholder
        for idx, probabilities in enumerate(class_probabilities):
            y[idx] = np.argmax(probabilities)
        return y

    def get_hidden(self,x):
        """
        You need to compute the z(value for the hidden layer; with tanh included) and retrieve it.
        """
        # extract the intermediate features computed at the hidden layers (after applying activation function)
        z = tanh(x@self.weight_1+self.bias_1)

        return z

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2
