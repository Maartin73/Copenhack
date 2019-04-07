#!/usr/bin/python3

import os
import cv2
import pandas as pd
import numpy as np
import skimage
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as k
from sklearn import preprocessing
from joblib import dump, load
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data():
    path_train = "./dataset/sign_mnist_train.csv"
    path_test = "./dataset/sign_mnist_test.csv"
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
    print(X_test[0])
    new_img = [x for img in X_test[0] for x in img]
    plt.imshow(new_img, cmap="gray")
    plt.show()

def main():
    #load_data()
    #train2()
    #train()
    #save()
    #dataset()
    eval()

imageSize=50
train_dir = "./dataset2/asl-alphabet/asl_alphabet_train/"
test_dir =  "./dataset2/asl-alphabet/asl_alphabet_test/"
from tqdm import tqdm

def get_data(folder):
    imageSize=50
    
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['A']:
                label = 0
            elif folderName in ['B']:
                label = 1
            elif folderName in ['C']:
                label = 2
            elif folderName in ['D']:
                label = 3
            elif folderName in ['E']:
                label = 4
            elif folderName in ['F']:
                label = 5
            elif folderName in ['G']:
                label = 6
            elif folderName in ['H']:
                label = 7
            elif folderName in ['I']:
                label = 8
            elif folderName in ['J']:
                label = 9
            elif folderName in ['K']:
                label = 10
            elif folderName in ['L']:
                label = 11
            elif folderName in ['M']:
                label = 12
            elif folderName in ['N']:
                label = 13
            elif folderName in ['O']:
                label = 14
            elif folderName in ['P']:
                label = 15
            elif folderName in ['Q']:
                label = 16
            elif folderName in ['R']:
                label = 17
            elif folderName in ['S']:
                label = 18
            elif folderName in ['T']:
                label = 19
            elif folderName in ['U']:
                label = 20
            elif folderName in ['V']:
                label = 21
            elif folderName in ['W']:
                label = 22
            elif folderName in ['X']:
                label = 23
            elif folderName in ['Y']:
                label = 24
            elif folderName in ['Z']:
                label = 25
            elif folderName in ['del']:
                label = 26
            elif folderName in ['nothing']:
                label = 27
            elif folderName in ['space']:
                label = 28           
            else:
                label = 29
            for image_filename in tqdm(os.listdir(folder + folderName)):
                img_file = cv2.imread(folder + folderName + '/' + image_filename,cv2.IMREAD_GRAYSCALE)
                if img_file is not None:       
                    img_file = skimage.transform.resize(img_file, (1,imageSize, imageSize))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y

def train2():
    global X_train, Y_train, X_test, Y_test

    train_dir = "dataset2/asl-alphabet/asl_alphabet_train/"
    test_dir =  "dataset2/asl-alphabet/asl_alphabet_test/"

    X_train, y_train = get_data(train_dir) 
    X_test, y_test= get_data(test_dir) # Too few images

    
    #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2) 

    # Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    
    Y_train = to_categorical(y_train, num_classes = 30)
    Y_test = to_categorical(y_test, num_classes = 30)
    # Shuffle data to permit further subsampling
    
    #X_train, Y_train = shuffle(X_train, Y_train, random_state=13)
    #X_test, Y_test = shuffle(X_test, Y_test, random_state=13)
    X_train = X_train[:30000]
    X_test = X_test[:30000]
    Y_train = Y_train[:30000]
    Y_test = Y_test[:30000]

def train():
    global model, X_train,Y_train, X_test, Y_test

    model = Sequential()
    k.set_image_dim_ordering('th')
    model.add(Convolution2D(64, 3, 3, border_mode= 'valid' , input_shape=(1, 50, 50), activation= 'relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation= 'relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation= 'relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation= 'relu' ))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation= 'softmax' ))
    # Compile model
    model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])

    # 9. Fit model on training data
    h = model.fit(X_train, Y_train, batch_size=128, epochs=10, verbose=1)
    
   

def eval():
    cnn = load('./trained_cnn.joblib')
    # 10. Evaluate model on test dat
    X_test, Y_test= get_data(test_dir)
    Y_test = to_categorical(Y_test, num_classes = 30)
    X_test = X_test[:30000]
    Y_test = Y_test[:30000]

    score = cnn.evaluate(X_test, Y_test, verbose=0)
    print(score)


def save():
    path = "./trained_cnn.joblib"
    dump(model,path)

main()