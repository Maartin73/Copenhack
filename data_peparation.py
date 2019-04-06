import cv2
import matplotlib.pyplot as plt

IMG_SIZE=28
    
#input: image path
def data_preparation(img):
    print(cv2.imread(img))
    img_array_gray=cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    print(img_array_gray)
    #plt.imshow(img_array_gray, cmap="gray")
    #plt.show()  
    new_img=cv2.resize(img_array_gray,(IMG_SIZE,IMG_SIZE))
    #plt.imshow(new_img, cmap="gray")
    #plt.show()
    return (new_img)

#output: image array 28x28 in gray scale
