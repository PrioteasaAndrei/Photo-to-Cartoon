from gaussian_blur import *
from util import *
import random
from sobel import *
from canny_edge_detector import *

'''

Imaginea citita cu cv2.imread e in format BGR

img[_][_][0] = BLUE
img[_][_][1] = GREEN
img[_][_][2] = RED


Edgeurile sunt negre la mine

255 = alb
0 = negru
'''
def main():
    lena_original = cv2.imread("img/lena_original.png")
    lena_grayscale = BGR_2_grayscale(lena_original)
    lena_blured = gaussian_blur(lena_grayscale,9)
    lena_sobel,angle_sobel = sobel_filters(lena_grayscale)
    lena_sobel = lena_sobel.astype(np.uint8)

    canny_edge_detector_img = canny_edge_detector(lena_original)
    print(canny_edge_detector_img)
    # show(np.concatenate((lena_sobel,canny_edge_detector_img), axis = 1))
   
   
    # show(lena_blured.astype(np.uint8))
    # show(lena_sobel.astype(np.uint8))
    # show(angle_sobel.astype(np.uint8))
    # show(np.concatenate((lena_sobel,lena_blured.astype(np.uint8)),axis=1))
if __name__ == "__main__":
    main()
