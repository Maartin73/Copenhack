#!/usr/bin/python3
import pandas as pd
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as k
from sklearn import preprocessing
from joblib import dump


import matplotlib.pyplot as plt


def load_data():
    path_train = "./dataset/sign_mnist_train.csv"
    path_test = "./dataset/sign_mnist_test.csv"

    global X_train, Y_train, X_test, Y_test

    #load the train data
    train = pd.read_csv(path_train).values
    X_train = train[:, 1:].reshape(train.shape[0],1,28, 28).astype( 'float32' )
    X_train = X_train / 255.0
    Y_train = train[:,0]
    lb = preprocessing.LabelBinarizer()
    Y_train = lb.fit_transform(Y_train)

    #load the test data
    test  = pd.read_csv(path_test).values
    X_test = test[:,1:].reshape(test.shape[0],1, 28, 28).astype( 'float32' )
    X_test = X_test / 255.0

    Y_test = test[:,0]
    Y_test = lb.fit_transform(Y_test)

    #d = numpy.resize(d,(27454,785))[1:]
    #x = [d[i][1:] for i in range(27451)] #llista de imatges
    #y = [d[i][0] for i in range(27451)] #llista de labels
    #y = numpy.list(y).astype('float32')
    #X_train = x.reshape(x.shape[0], 1, 28, 28)
    #X_train = X_train.astype('float32') 
    #X_train /= 255
    #print(X_train)


def dataset():
    #print(X_test[0])
    new_img = [x for img in X_test[0] for x in img]
    plt.imshow(new_img, cmap="gray")
    plt.show()
def main():
    load_data()
    #train()
    #save()
    dataset()

def train():
    global model

    model = Sequential()
    k.set_image_dim_ordering('th')
    model.add(Convolution2D(30, 5, 5, border_mode= 'valid' , input_shape=(1, 28, 28),activation= 'relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(15, 3, 3, activation= 'relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation= 'relu' ))
    model.add(Dense(50, activation= 'relu' ))
    model.add(Dense(24, activation= 'softmax' ))
    # Compile model
    model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])

    # 9. Fit model on training data
    h = model.fit(X_train, Y_train, batch_size=16, epochs=5, verbose=1)
    
    # 10. Evaluate model on test data
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)


def save():
    path = "./trained_cnn.joblib"
    dump(model,path)

main()