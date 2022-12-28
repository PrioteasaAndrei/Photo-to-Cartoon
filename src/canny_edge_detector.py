import numpy as np
from util import *
from gaussian_blur import *
from sobel import *

def non_maximum_supression(G,theta):
    ret = np.zeros_like(G)

    for i in range(1,G.shape[0] - 1):
        for j in range(1,G.shape[1] - 1):
            try:
                neigh1 = 255
                neigh2 = 255

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

    return ret



'''
Every pixel with intensity higher than highTreshold becomes white, weak becomes gray 25, others become black 0
'''
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


def edge_tracking(img, weak, strong=255):
    M, N = img.shape  

    ## track in 3x3 neighbourhood
    offset_x = [1,1,1,0,0,-1,-1,-1]
    offset_y = [-1,0,1,-1,1,-1,0,1]

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    neighbours = [img[i+ox,j+oy] for ox,oy in zip(offset_x,offset_y)]
                    if len(list(filter(lambda x: x == strong,neighbours))) > 0:
                        img[i,j] = strong
                    else:
                        img[i,j] = 0

                except IndexError as e:
                    pass
    return img


def canny_edge_detector(img):
    grayscale_img = BGR_2_grayscale(img)
    blured_img = gaussian_blur(grayscale_img,9)
    amplitude,angle = sobel_filters(blured_img)
    after_non_max_sup = non_maximum_supression(amplitude,angle)
    after_treshold_res,after_treshold_weak,after_treshold_strong = threshold(after_non_max_sup)
    after_edge_tracking = edge_tracking(after_treshold_res.copy(),25)

    return after_non_max_sup,after_treshold_res,after_edge_tracking