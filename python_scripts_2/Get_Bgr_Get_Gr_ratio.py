import cv2
import numpy as np
import show_image
from show_image import *


def get_BGR(image,cx,cy):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    #cnt = np.array([[[cx + 20, cy + 20]], [[cx - 20, cy + 20]], [[cx - 20, cy - 20]], [[cx + 20, cy - 20]]])
    cnt1 = np.array([[[cx + 15, cy + 15]], [[cx - 15, cy + 15]], [[cx - 15, cy - 15]], [[cx + 15, cy - 15]]])
    cv2.drawContours(mask, [cnt1], 0, 255, -1)
    [mean_blue, mean_green, mean_red, _] = map(int, cv2.mean(image, mask=mask))
    #cv2.drawContours(image, [cnt1], 0, 255, 5)
    #show_Image(image,"output_image")
    return[mean_red, mean_green,mean_blue]

def get_Red_Green_Ratio(image,cx,cy) :
    [mean_red, mean_green,mean_blue] =get_BGR(image,cx,cy)
    ratio=mean_red/mean_green
    return  ratio