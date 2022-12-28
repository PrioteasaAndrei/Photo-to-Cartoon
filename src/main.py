from gaussian_blur import *
from util import *
import random
from sobel import *
from canny_edge_detector import *
from color_quantization import *
from median_filter import *

'''
Imaginea citita cu cv2.imread e in format BGR

img[_][_][0] = BLUE
img[_][_][1] = GREEN
img[_][_][2] = RED

Edgeurile sunt negre la mine

255 = alb
0 = negru

pentru afisarea cu plt trebuie sa convertesti BGR la RGB 
'''
def main():
    lena_original = cv2.imread("../img/lena_original.png")
    lena_grayscale = BGR_2_grayscale(lena_original)
    lena_blured = gaussian_blur(lena_grayscale,9)
    lena_sobel,angle_sobel = sobel_filters(lena_blured)
    after_non_max_sup,after_treshold_res,after_edge_tracking = canny_edge_detector(lena_original)

    first_row = np.concatenate((lena_grayscale,lena_blured,lena_sobel,angle_sobel),axis=1)
    second_row = np.concatenate((after_non_max_sup,after_treshold_res,after_edge_tracking,np.ones_like(angle_sobel) * 255),axis=1)
    combined = np.concatenate((first_row,second_row),axis=0)

    show_plt(combined,"First row: Grayscale     Gaussian Blur       Sobel Amplitudes        Sobel quantized angles\nSecond row: Non maximum supression      Threshold hysteresis        Edge tracking")

    save_image(lena_grayscale,"../img/lena_grayscale")
    save_image(lena_blured,"../img/lena_blured")
    save_image(lena_sobel,"../img/lena_sobel")
    save_image(angle_sobel,"../img/angle_sobel")
    save_image(after_non_max_sup,"../img/non_max_supression")
    save_image(after_treshold_res,"../img/tresholding")
    save_image(after_edge_tracking,"../img/edge_tracking")

def test_k_means():
    lena_original = cv2.imread("../img/lena_original.png")
    color_quant_k_means = color_quantization(lena_original,20)
      # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    show_plt(cv2.cvtColor(color_quant_k_means, cv2.COLOR_BGR2RGB))

def test_k_means_median():
    lena_original = cv2.imread("../img/lena_original.png")
    color_quant_median_wrapper(lena_original,4)

def test_median_filter():
    lena_original = cv2.imread("../img/lena_original.png")
    lena_color_quant = color_quant_median_wrapper(lena_original,4)
    # show_plt(lena_color_quant)
    save_image(np.concatenate((lena_color_quant, median_filter(lena_color_quant,kernel_size=5), median_filter(lena_color_quant,kernel_size=7)),axis=1),"../img/median_filter_0_5_7")
    

def test_combine_images():
    lena_original = cv2.imread("../img/lena_original.png")
    lena_color_quant = color_quant_median_wrapper(lena_original,4)
    lena_median = median_filter(lena_color_quant,kernel_size=5)
    after_non_max_sup,after_treshold_res,after_edge_tracking = canny_edge_detector(lena_original)

    final = combine_img_with_canny(lena_median,after_edge_tracking)
    save_image(final,"../img/combined")
    show_plt(final)

if __name__ == "__main__":
    # main()
    # test_k_means_median()
    # test_k_means_median()
    # test_median_filter()
    test_combine_images()