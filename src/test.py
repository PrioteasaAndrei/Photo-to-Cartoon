import numpy as np
import cv2

img  = cv2.imread("../img/lena_original.png")


flattened_img= np.array([[img[i,j,2],img[i,j,1],img[i,j,0],i,j] for i in range(img.shape[0]) for j in range(img.shape[1])])

print(flattened_img)

## prima coloana
print(flattened_img[:,0])

## ultima coloana
print(flattened_img[:,4])
