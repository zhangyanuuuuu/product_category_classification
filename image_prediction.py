import numpy as np


def knn_finder(x_train, y_train, x_test):
    sim = np.matmul(x_train, x_test.T)
    max_index = np.argmax(sim)
    return y_train[max_index]


def predict_image_label():
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_test = np.load('x_test.npy')

    y_pred = np.zeros(x_test.shape[0])
    for i, row in enumerate(x_test):
        y_pred[i] = knn_finder(x_train, y_train, row)

    with open('y_image_pred.npy', 'wb') as f:
        np.save(f, y_pred)
