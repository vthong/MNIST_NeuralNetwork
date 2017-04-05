from struct import *
import numpy as np
import matplotlib.pyplot as plt
import neuralNet as nn
import pickle as pc
import cv2

# TEST
# t10k-images.idx3-ubyte
# t10k-labels.idx1-ubyte

# TRAIN
# train-images.idx3-ubyte
# train-labels.idx1-ubyte

def image_data(data):
    """ Convert data to numpy 2D array. A image is a column. """
    # '>', value in big-endian format
    # 'i', extract 4 byte to int value
    # 'B', extract 1 byte to unsigned char value
    num_of_image = unpack('>i', data[4:8])[0]
    num_of_row = unpack('>i', data[8:12])[0]
    num_of_col = unpack('>i', data[12:16])[0]

    format_str = '>{:d}B'.format(num_of_image * num_of_row * num_of_col)
    pixel = unpack_from(format_str, data, offset=16)

    # Number of variable for a image is: num_of_row*num_of_col
    # Number of observation is: num_of_image
    # Each column is a observaton
    ls_img = np.reshape(pixel, (num_of_col * num_of_row, num_of_image),\
                        order='F')
    return ls_img

def data_label(data):
    """ Convert Label data in data to numpy 1D array. """
    # '>', value in big-endian format
    # 'i', extract 4 byte to int value
    # 'B', extract 1 byte to unsigned char value
    num_of_item = unpack('>i', data[4:8])[0]

    format_str = '>{:d}B'.format(num_of_item)
    labels = unpack_from(format_str, data, offset=8)

    # Convert to numpy 1D array
    ls_label = np.ravel(labels)
    return ls_label

def extract_pca(data, nComponent = 784):
    """ Extract PCA"""
    data_mean = np.mean(data, axis=1).reshape(-1,1)
    # Calculate the covariance matrix of data
    # rowvar = True: a column is a sample
    data_cov = np.cov(data, rowvar=True)
    # Calculate eigenvectors and eigenvalue
    # eig_vecs[:, i] corresponding with eig_vals[i]
    eig_vals, eig_vecs = np.linalg.eigh(data_cov)
    # Sort decrease eigenvector
    sort_idx = np.argsort(eig_vals)
    sort_idx = sort_idx[::-1]
    eig_vecs = eig_vecs[:, sort_idx]
    # Keep nComponent eigenvector corresponding nComponent-largest eigenvalue
    nC = nComponent if nComponent > 0 and nComponent <= eig_vecs.shape[0]\
        else eig_vecs.shape[0]
    eig_vecs_nComponent = eig_vecs[:, 0:nComponent]
    # Extract nComponent of data
    transform = eig_vecs_nComponent.T.dot(data-data_mean)
    return eig_vecs_nComponent, transform

if __name__ == "__main__":

    # Read data from Files
    # Data file is from http://yann.lecun.com/exdb/mnist/
    # And extract files gz
    fTrain = open("train-images-idx3-ubyte", mode="rb")
    fLabel = open("train-labels-idx1-ubyte", mode="rb")

    fTest = open("t10k-images-idx3-ubyte", mode="rb")
    fTestLabel = open("t10k-labels-idx1-ubyte", mode="rb")

    print("Start Reading data...")
    try:
        imgTrain = fTrain.read();
        trainLabel = fLabel.read();

        imgTest = fTest.read();
        testLabel = fTestLabel.read();
    finally:
        fTrain.close()
        fLabel.close()
        fTest.close()
        fTestLabel.close()

    # End Read Data from files

    # Current system is little-endian
    # Binary data in files read is big-endian
    # '>' Read as big-endian

    # Split imgTrain to array

    lsImg = image_data(imgTrain)
    lsLabel = data_label(trainLabel)

    lsImgTest = image_data(imgTest)
    lsLabelTest = data_label(testLabel)

    print("End read data.\n")

    nC = 225;
    eig_vec_nComponent, X = extract_pca(lsImg, nComponent=nC)
    xTest = eig_vec_nComponent.T.dot(lsImgTest-np.mean(lsImg, axis=1).reshape(-1,1))

    layers = [np.asarray([nC, 500, 500, 10], dtype=np.uint16)]
    layers.append(np.asarray([nC, 500, 10], dtype=np.uint16))
    learning_rate=1e-3
    epoch=50
    batch_size=200

    #for i in range(1):
        #model = nn.NeuralNetwork(layers[0], learning_rate, epoch, batch_size)
        # Train model
        #model.fit(X, lsLabel)

        # Predict on Test
        #loss, correct = model.predict(xTest, lsLabelTest)
        #model.write_csv([layers[0], learning_rate, epoch, batch_size, loss, correct], "test_correct.csv")
        
        #Save Model to neuralNet.model
        #with open("neuralNet.model", "wb")  as output:
            #pc.dump(model, output)
        
        #model2 = nn.NeuralNetwork(layers[1], learning_rate, epoch, batch_size)
        #model2.fit(X, lsLabel)

        #loss, correct = model2.predict(xTest, lsLabelTest)
        #model.write_csv([layers[1], learning_rate, epoch, batch_size, loss, correct], "test_correct.csv")

    # Load Model
    with open("neuralNet.model", "rb") as fileIn:
        model = pc.load(fileIn)
    
    model.predict(xTest, lsLabelTest)
    #Plot Image
    plt.title('Label is {label}'.format(label=lsLabelTest[1000]))
    plt.imshow(lsImgTest[:, 1000].reshape(28,28), cmap='gray')
    plt.show()
    # End Plot Image