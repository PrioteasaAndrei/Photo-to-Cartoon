import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image
import os

def convolution(image, kernel, average=False, verbose=False):

    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = BGR_2_grayscale(image)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

    print("Kernel Shape : {}".format(kernel.shape))

    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    print("Output Image size : {}".format(output.shape))

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()

    return output



# ## RGB
# [0.2989, 0.5870, 0.1140]

# ## BGR
# [0.1140,0.5870,0.2989]
def BGR_2_grayscale(img):
    tmp = img.copy()
    return np.dot(tmp[...,:3], [0.1140,0.5870,0.2989])


def show(img,keep=True,tag="Default"):
    cv2.imshow(tag,img)
    if keep:
        cv2.waitKey(0)
    # cv2.destroyAllWindows()


def show(imgs,no_of_images=2,tag="default"):
    to_tuple = tuple(imgs)
    cv2.imshow(tag,np.concatenate(to_tuple,axis=1))
    cv2.waitKey(0)


def show_plt(img,title="Default"):
    plt.imshow(img, interpolation='none', cmap='gray')
    plt.title(title)
    plt.show()


def save_image(img,path,overwrite=False):
    im = Image.fromarray(img)
    if im.mode != 'RGB':
        im = im.convert('RGB')

    if os.path.isfile(path + ".png") and overwrite==False:
        print("File already exist in directory. Set overwrite flag to True to overwrite it.")
        return
    else:
        im.save(path + ".png")
        print("Saved file " + path + " to directory.")

    

def combine_img_with_canny(img,img_canny):
    out_img = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img_canny[i,j] == 0:
                out_img[i,j] = img[i,j]
            else:
                out_img[i,j] = img_canny[i,j] - 255

    return out_img