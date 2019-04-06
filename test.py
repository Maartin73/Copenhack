#!/usr/bin/python3
from joblib import load
import cv2
import matplotlib.pyplot as plt

IMG_SIZE=28
    
#input: image path
def load_img(img):
    #print(cv2.imread(img))
    img_array_gray=cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    #print(img_array_gray)
    new_img=cv2.resize(img_array_gray,(IMG_SIZE,IMG_SIZE))
    plt.imshow(new_img, cmap="gray")
    plt.show()
    return (new_img)
#output: image array 28x28 in gray scale



def test(img):
    #load cnn
    model = load('./trained_cnn.joblib')
    #test
    model.predict(load_img(img))

def main():
    x= load_img('./testset/1.jpg')
    print(x)
main()