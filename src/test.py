import numpy as np
import cv2
from util import *

img  = cv2.imread("../img/lena_original.png")
gray = BGR_2_grayscale(img)

# flattened_img= np.array([[img[i,j,2],img[i,j,1],img[i,j,0],i,j] for i in range(img.shape[0]) for j in range(img.shape[1])])

# print(flattened_img)

# ## prima coloana
# print(flattened_img[:,0])

# ## ultima coloana
# print(flattened_img[:,4])

i = 3
j = 4
kernel = 3
lista = list(gray[i - 1: i + 2,j-1 : j+2])
ret = []
for el in lista:
    for _el in el:
        ret.append(_el)

print(gray[i - 1: i + 2,j-1 : j+2],gray[i,j],np.median(gray[i - 1: i + 2,j-1 : j+2]))