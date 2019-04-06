import cv2;
import show_image
from show_image import *
import custom_contour_py
from custom_contour_py import *
import numpy as np

def get_the_contours(colorchecker,gray, fast1,fast2,Gb1, Gb2,it1,iterations,cn1,cn2):
    colorchecker_area=np.shape(colorchecker)[0]*np.shape(colorchecker)[1]
    colorchecker_lower_ratio=colorchecker_area/172
    colorchecker_upper_ratio=colorchecker_area/11
    print(colorchecker_area)

    colorchecker_blurred = cv2.fastNlMeansDenoising(gray, 10, fast1,fast2, 21)
    colorchecker_blurred = cv2.GaussianBlur(colorchecker_blurred, (Gb1, Gb2), 0)
    colorchecker_canny = cv2.Canny(colorchecker_blurred, cn1,cn2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    colorchecker_dilation = cv2.dilate(colorchecker_canny, kernel, iterations=it1)
    colorchecker_closed = cv2.morphologyEx(colorchecker_dilation, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    (_, cnts, hie) = cv2.findContours(colorchecker_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   # show_Image(colorchecker, "color_checker")
    #show_Image(gray, "gray")
   # show_Image(colorchecker_canny, "canny")
    #show_Image(colorchecker_closed, "closed")
    index = 0
    hie = hie[0]

    candidates = []
    contours_by_index = {}
    for component in zip(cnts, hie):
        con = CustomContour(index, component[0], component[1])
        contours_by_index[index] = con

        if con.area > colorchecker_lower_ratio and con.area < colorchecker_upper_ratio and con.cX is not None:
            candidates.append(con)

            index += 1
    print(len(candidates))
    return cnts, candidates