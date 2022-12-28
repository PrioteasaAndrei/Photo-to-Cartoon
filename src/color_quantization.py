from sklearn.cluster import KMeans
import cv2
import numpy as np
from util import *
import math 

def color_quantization(img, K):
# Defining input data for clustering
  data = np.float32(img).reshape((-1, 3))
# Defining criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
# Applying cv2.kmeans function
  ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(img.shape)
  return result



'''
list_of_px = [[]]
'''
def get_color_mean(list_of_px):
  sum_px = np.array([0,0,0],dtype=float)
  for px in list_of_px:
    sum_px += px
  
  return sum_px / len(list_of_px)


def closest_color(px, list_of_quant_colors):
  
  ret = None
  mind = 1000000 
  for i in range(len(list_of_quant_colors)):
    curr_quant = list_of_quant_colors[i]
    ## euclidian distance
    d = math.sqrt((int(curr_quant[0]) - int(px[0])) ** 2 + (int(curr_quant[1]) - int(px[1])) ** 2 + (int(curr_quant[2]) - int(px[2])) ** 2)
    if d < mind:
      mind = d
      ret = curr_quant

  return ret


'''
Imaginea e BGR
'''
def color_quant_median(img,splits=4):
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  ## acounted for the inverse order
  red = img[:, :, 2]
  green = img[:, :, 1]
  blue = img[:, :,0]

  ## RGB order
  img_red = np.stack([red, np.zeros_like(red), np.zeros_like(red)], axis=-1)  
  img_blue = np.stack([np.zeros_like(blue), np.zeros_like(blue),blue], axis=-1)  
  img_green = np.stack([np.zeros_like(green),green, np.zeros_like(green)], axis=-1)  

  # show_plt(img)
  # show_plt(img_red)
  # show_plt(img_blue)
  # show_plt(img_green)


  max_diff = -1
  max_channel = np.zeros_like(red)
  max_channel_str = 'I'

  if red.max() - red.min() > max_diff:
    max_diff = red.max() - red.min()
    max_channel = red
    max_channel_str = 'r'
  
  
  if green.max() - green.min() > max_diff:
    max_diff = green.max() - green.min()
    max_channel = green
    max_channel_str = 'g'

  
  if blue.max() - blue.min() > max_diff:
    max_diff = blue.max() - blue.min()
    max_channel = blue
    max_channel_str = 'b'

  
  # if max_channel == np.zeros_like(red):
  #   raise Exception()

  print(max_channel_str)
  cv2.waitKey(0)
  ## RGB order
  bucket = [(img[i,j,2],img[i,j,1],img[i,j,0]) for i in range(img.shape[0]) for j in range(img.shape[1])]

  if max_channel_str == 'r':
    bucket = sorted(bucket,key = lambda x: x[0])
  elif max_channel_str == 'g':
    bucket = sorted(bucket,key = lambda x: x[1])
  elif max_channel_str == 'b':
    bucket = sorted(bucket,key = lambda x: x[2])


  # print(list(map(lambda x: x[1],bucket)))
  chunked_np_arrays = np.array_split(np.array(bucket),2 ** splits)
  chunked_array = [list(arr) for arr in chunked_np_arrays]
  chunked_array_means = list(map(get_color_mean,chunked_array))
  chunked_array_means = [x.astype(np.uint8) for x in chunked_array_means]
  ## chunked_array[i][j] = np.array acum


  # print(chunked_array_means)
  # print(chunked_array_means[0].astype(np.uint8))
  
  bitmap = np.zeros_like(img)

  ## bitmap va contine pt ficare pixel indicele bucketului din care face parte 
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      current_px = img[i,j]
      bitmap[i,j] = closest_color(img[i,j],chunked_array_means)


  show_plt(bitmap)

'''
Tine i si j pt fiecare pixel in tuplu
'''

def color_quant_median_wrapper(img,splits):
  ## RGB format
  out_img = np.zeros_like(img)

  ## bucket has RGB ordeer
  def color_quant_median_rec(img,bucket,splits):

    if len(bucket) == 0:
      return
    
    if splits == 0:
      # print(bucket)
      r_average = np.mean(bucket[:,0])
      g_average = np.mean(bucket[:,1])
      b_average = np.mean(bucket[:,2])

      for el in bucket:
        out_img[el[3],el[4]] = [r_average,g_average,b_average]

      return

    ## acounted for the inverse order
    print("TYPE",type(bucket))
    red = bucket[:,0]
    green = bucket[:,1]
    blue = bucket[:,2]

    max_diff = -1
    max_channel_str = 'I'
    color_index = -1

    if red.max() - red.min() > max_diff:
      max_diff = red.max() - red.min()
      max_channel_str = 'r'
      color_index = 0
    
    
    if green.max() - green.min() > max_diff:
      max_diff = green.max() - green.min()
      max_channel_str = 'g'
      color_index = 1

    
    if blue.max() - blue.min() > max_diff:
      max_diff = blue.max() - blue.min()
      max_channel_str = 'b'
      color_index = 2

    bucket = bucket[bucket[:,color_index].argsort()]

    color_quant_median_rec(img,bucket[: int(len(bucket) / 2) ],splits - 1)
    color_quant_median_rec(img,bucket[int(len(bucket) / 2):],splits - 1)


  
  flattened_img= np.array([[img[i,j,2],img[i,j,1],img[i,j,0],i,j] for i in range(img.shape[0]) for j in range(img.shape[1])])
  color_quant_median_rec(img,flattened_img ,splits)

  # show_plt(out_img)
  return out_img