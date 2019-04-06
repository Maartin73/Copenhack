#!/usr/bin/python3
from joblib import load
import cv2
import matplotlib.pyplot as plt

IMG_SIZE=28
    
#input: image path
def load_img(img):
    #print(cv2.imread(img))
    img_array_gray=cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    
    new_img=cv2.resize(img_array_gray,(IMG_SIZE,IMG_SIZE))
    new_img=cv2.normalize(new_img,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)

    #plt.imshow(new_img, cmap="gray")
    #plt.show()
    new_img = new_img.reshape(1,1,28,28)
    print(new_img)
    return new_img
    
    
def test(img):
    #load cnn
    model = load('./trained_cnn.joblib')
    #test
    score = model.predict(load_img(img))
    print("Hola:\n")
    print(score)

def main():
    test('./testset/2.jpeg')
    
main()