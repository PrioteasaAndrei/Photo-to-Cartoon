import numpy as np
from util import *

def sobel_filters(img,to_degrees=True,_range_conversion=True,gradient_quantization=True):
    _img = img.copy()
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = convolution(_img, Kx)
    Iy = convolution(_img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    

    ## reverse white and black

    # G = 255 * np.ones_like(G) - G

    ## convert to degrees
    ## theta between -180 and 180 degrees

    if to_degrees:
        theta[:] = np.degrees(theta[:])

    if _range_conversion:
        theta = range_conversion(theta)

    if gradient_quantization:
        vfunc = np.vectorize(quantization_function)
        theta = vfunc(theta)


    return (G, theta)



def thicken_edges(img):
    pass

def quantization_function(theta):
    ret = None

    if (theta >= 0 and theta <= 22.5) or (theta >= 337.5 and theta <= 360) or (theta >= 157.5 and theta <= 202.5):
        ret = 0
    elif (theta >= 22.5 and theta <= 67.5) or (theta >= 202.5 and theta <= 247.5):
        ret = 45
    elif (theta >= 67.5 and theta <= 112.5) or (theta >= 247.5 and theta <= 292.5):
        ret = 90
    elif (theta >= 112.5 and theta <= 157.5) or (theta >= 292.5 and theta <= 337.5):
        ret = 135

    if ret == None:
        raise Exception()

    return ret
        


## converts from -180 180 to 0 360
def range_conversion(theta):
    ret = theta.copy()
    return (ret + 360) % 360

