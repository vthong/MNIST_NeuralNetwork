import numpy as np
import activation as act
from timeit import default_timer as timer
import csv

class NeuralNetwork:
    """ Implement Neural Network - Backpropagation  """
    def __init__(self, layers, learning_rate=0.001, epoch=20, batch_size=100):
        """ Init Paramater
        learning_rate: learning rate coefficient
        epoch: number of epoch
        layers[]: number of neural at a layer 
        batch_size: number of sample in a batch
        W[i, j, k]: weights from neural j of layer i-1 to neural k of layer i
            size of W: nLayer-1 x nNeuron_i+1 x nNeuron_j
        """

        self.nLayer = len(layers)
        self.W = []
        np.random.seed(1)
        for i in range(self.nLayer - 1):
            r = np.sqrt(6. / (layers[i] + 1 + layers[i + 1]))
            wi = np.random.uniform(-r, r, (layers[i] + 1, layers[i + 1]))
            self.W.append(wi)

        self.lr = learning_rate
        self.epoch = epoch
        self.batchSize = batch_size
        self.activation = act.tanh

    def batch_train(self, data, label):
        """ Batch Training 
        X[i, j, k]: input for layer i
            size of X: nLayer x nNeuron_i x nSample
        D[i, :]: delta for layer i (except input layer)
            size of D: nLayer-1 x nNeuron_i x nSample
        Fd[i, :]: derivative of activation of layer i 
            (except input layer and output layer)
            size of Fd: nLayer-2 x nNeuron_i x nSample
        """
        n_samples = data.shape[1]
        # Add bias unit to input layer
        bias = np.ones((1, n_samples))
        X = [np.concatenate((data, bias), axis=0)]
        Fd = []
        # Forward
        for i in range(self.nLayer - 2):
            si = np.dot(self.W[i].T, X[i])
            xi = self.activation(si)
            xi_deriv = self.activation(si, 1)
            xi = np.concatenate((xi, bias), axis=0)
            X.append(xi)
            Fd.append(xi_deriv)
        so = np.dot(self.W[-1].T, X[-1])
        xo = act.softmax(so)
        # Backpropagation
        o_delta = xo
        o_delta[label, np.arange(n_samples)] -= 1
        D = [o_delta]
        for i in range(self.nLayer - 3, -1, -1):
            delta = self.W[i + 1][0:-1,:].dot(D[-1])
            delta = np.multiply(delta, Fd[i])
            D.append(delta)
        D.reverse()
        # Update weight
        for i in range(self.nLayer - 1):
            self.W[i] += (-self.lr * X[i].dot(D[i].T))
        #Release Memory
        D.clear()
        Fd.clear()
        X.clear()

    def shuffle_data(self, X, y):
        """ Shuffle [X, y] """
        data = np.concatenate((X.T, y.reshape(-1,1)), axis = 1)
        np.random.shuffle(data)
        return data[:, 0:-1].T, np.uint8(np.ravel(data[:, -1]))

    def fit(self, X, y):
        """Training base on training set X, label y
        X: numpy array size [n_feature, n_sample]
        y: numpy array size [n_sample]
        """
        cross_entropy_loss = []
        num_correct = []
        start_fit = timer()
        print("Start Fitting:...", end=". ")
        for i in range(self.epoch):
            X, y = self.shuffle_data(X, y)
            start = timer()
            n_samples = X.shape[1]
            n_batch = np.ceil(n_samples / self.batchSize)
            for k in range(np.uint16(n_batch)):
                end_batch = n_samples if (k + 1) * self.batchSize >= n_samples \
                            else (k + 1) * self.batchSize
                Xb = X[:, k * self.batchSize: end_batch]
                yb = y[k * self.batchSize: end_batch]
                self.batch_train(Xb, yb)
            loss, n_correct = self.predict(X, y)
            cross_entropy_loss.append(loss)
            num_correct.append(n_correct)
        print("Total Fit Time: ", timer()-start_fit)
        # Write to csv file
        self.write_csv(cross_entropy_loss, "train.csv")
        self.write_csv(num_correct, "train_correct.csv")

    def predict(self, X, y=None):
        """Preditc Label for X"""

        p_label = np.empty((0,0))
        n_samples = X.shape[1]
        n_batch = np.ceil(n_samples / self.batchSize)
        loss = 0
        for i in range(np.uint16(n_batch)):
            end_batch = n_samples if (i + 1) * self.batchSize >= n_samples else (i + 1) * self.batchSize
            cur_batch = end_batch - i * self.batchSize
            Xb = X[:, i * self.batchSize: end_batch]
            # Add bias
            bias = np.ones((1, cur_batch))
            Xb = np.concatenate((Xb, bias), axis=0)
            # Forward
            for k in range(self.nLayer - 2):
                sk = np.dot(self.W[k].T, Xb)
                Xb = self.activation(sk)
                Xb = np.concatenate((Xb, bias), axis=0)
            so = np.dot(self.W[-1].T, Xb)
            Xb = act.softmax(so)
            if y is not None:
                loss += self.loss_function(Xb, y[i * self.batchSize: end_batch])
            p_label = np.append(p_label, np.argmax(Xb, axis=0))
        loss = loss / n_samples

        if y is not None:
            cnt = np.sum( y == p_label)
            correct = (cnt * 10000 // len(y)) / 100
            print("Loss, Correct: ", loss, correct)
            return loss, correct
        else:
            print("Label: ", p_label)
            return None, None

    def loss_function(self, y_pred, target):
        """ Calculate cross entropy error. 
        y_pred: numpy 2D array shape = (10, n_samples)
        target: numpy 1D array shape = (1, n_samples)
        return values: float, cross entropy error
        """
        n_samples = y_pred.shape[1]
        target_2d = np.zeros((10, n_samples))
        target_2d[target, np.arange(n_samples)] = 1
        entropy_err = np.multiply(target_2d, np.log(y_pred)) + \
                        np.multiply(1 - target_2d, np.log(1 - y_pred))
        return (-1) * np.sum(entropy_err)

    def write_csv(self, data, fname):
        '''Write data to csv file'''
        file = open(fname, 'a', newline='')
        fwriter = csv.writer(file, delimiter=',')
        fwriter.writerow(data)
        file.close()