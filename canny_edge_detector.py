import numpy as np
from util import *
from gaussian_blur import *
from sobel import *

## TODO: nu pot sa deosebesc 180 de 0 si nu stiu cum sa iau vecinii
def non_maximum_supression(G,theta):
    ret = np.zeros_like(G)

    for i in range(1,G.shape[0] - 1):
        for j in range(1,G.shape[1] - 1):
            try:
                neigh1 = 0
                neigh2 = 0

                if theta[i,j] == 0:
                    neigh1 = G[i,j+1]
                    neigh2 = G[i,j-1]
                elif theta[i,j] == 45:
                    neigh1 = G[i+1,j-1]
                    neigh2 = G[i-1,j+1]
                elif theta[i,j] == 90:
                    neigh1 = G[i+1,j]
                    neigh2 = G[i-1,j]
                elif theta[i,j] == 135:
                    neigh1 = G[i-1,j-1]
                    neigh2 = G[i+1,j+1]


                if (G[i,j] >= neigh1) and (G[i,j] >= neigh2):
                    ret[i,j] = G[i,j]
                else:
                    ret[i,j] = 0
            except IndexError as e:
                pass 

    return np.ones_like(ret) * 255 -  ret


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def canny_edge_detector(img):
    grayscale_img = BGR_2_grayscale(img)
    blured_img = gaussian_blur(grayscale_img,9)
    amplitude,angle = sobel_filters(blured_img)
    after_non_max_sup = non_maximum_supression(amplitude,angle)
    after_treshold_res,after_treshold_weak,after_treshold_strong = threshold(after_non_max_sup)


    show(np.concatenate((after_non_max_sup.astype(np.uint8),after_treshold_res.astype(np.uint8)),axis=1),tag="Left: Non max supression    Right: Double treshold")


    return after_non_max_sup.astype(np.uint8)