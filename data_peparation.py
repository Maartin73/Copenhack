import cv2
import matplotlib.pyplot as plt

IMG_SIZE=28

def main():
    img_path="./descarga.jpg"
    data_preparation(img_path)
    

def data_preparation(img):
    print(cv2.imread(img))
    img_array_gray=cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    print(img_array_gray)
    #plt.imshow(img_array_gray, cmap="gray")
    #plt.show()  
    new_img=cv2.resize(img_array_gray,(IMG_SIZE,IMG_SIZE))
    #plt.imshow(new_img, cmap="gray")
    #plt.show()

    img_ret=[]
    id1=0
    id2=0
    for row in new_img:
        id2=0
        rowlist=[]
        for pos in row:
            rowlist.append(new_img[id1,id2]/255.0)
            id2+=1
        id1+=1
        img_ret.append(rowlist)

    print(img_ret)


main()
