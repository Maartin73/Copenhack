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

IMG_SIZE=50
    
#input: image path
def load_img(img):
    #print(cv2.imread(img))
#img_file = cv2.imread(folder + folderName + '/' + image_filename,cv2.IMREAD_GRAYSCALE)
 #               if img_file is not None:       
  #                  img_file = skimage.transform.resize(img_file, (1,imageSize, imageSize))
   #                 img_arr = np.asarray(img_file)

    img_array_gray=cv2.imread(img, cv2.IMREAD_GRAYSCALE) 
    print(img_array_gray)
    new_img = skimage.transform.resize(img_array_gray, (1,IMG_SIZE, IMG_SIZE))
    print(new_img)
    img_array = np.asarray(new_img)

 #   new_img=cv2.resize(img_array_gray,(IMG_SIZE,IMG_SIZE))
  #  new_img=cv2.normalize(new_img,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)

    #plt.imshow(new_img, cmap="gray")
    #plt.show()
    new_img = img_array.reshape(1,1,50,50)
    print(new_img)
    return new_img
    
    
def test(img):
    #load cnn
    model = load('./trained_cnn.joblib')
    imgin=load_img(img)
    #test
    score = model.predict_classes(imgin)
    print("Hola:\n")
    print(score)
    score2 = model.predict(imgin)
    print(score2)

def main():
    test('./testset/N1909.jpg')
    
main()