import numpy as np

'''
img is in RGB order
'''
def median_filter(img,kernel_size=3):

    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :,2]

    median_red = median_filter_inner(red,kernel_size)
    print("Done red")
    # print(median_red)
    median_green = median_filter_inner(green,kernel_size)
    print("Done green")
    # print(median_green)
    median_blue = median_filter_inner(blue,kernel_size)
    print("Done blue")
    # print(median_blue)

    ret = np.dstack((median_red,median_green,median_blue))
    print(ret.shape)

    return ret

def median_filter_inner(img,kernel_size=3):
    out_img = np.zeros_like(img)

    ## ignore padding 

    # print(img.shape)
    image_row, image_col = img.shape
    kernel_row, kernel_col = kernel_size,kernel_size

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = img

    lower_bound = kernel_size // 2
    upper_bound = kernel_size // 2 + 1 

    for row in range(image_row):
        for col in range(image_col):
            row_padded = row + pad_height
            col_padded = col + pad_width

            cut = padded_image[row_padded-lower_bound:row_padded+upper_bound,col_padded-lower_bound:col_padded+upper_bound]
            out_img[row,col] = np.median(cut)


    return out_img