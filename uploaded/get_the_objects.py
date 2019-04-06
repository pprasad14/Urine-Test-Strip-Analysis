#image_path="IMG-20180814-WA0002.jpg"
import os
import cv2
import show_image
from show_image import *
import numpy as np
import get_Cropped_image
from get_Cropped_image import *
import logging as log


def get_objects(image,gb1,gb2,cn1,cn2,it) :


    # show_Image(image,"original_image")

    # Apply gaussian blur to the images
    blurred = cv2.GaussianBlur(image, (gb1,gb2), 0)

    # get the edges of the image    # show the edged image
    edged = cv2.Canny(blurred, cn1, cn2)

    #show_Image(edged, "Canny_image")

    # Get the area of the Image andd contour ratio
    Area = np.shape(image)[0] * np.shape(image)[1]
    lower_Ratio = 400

    # get the Kernel for closing the edges to get the contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=it)
    #show_Image(closed, "closed")

    # get the contours and sort it based on the area
    try:
        (_, cnts, hie) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        (cnts, hie) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if hie is None:
        return [],0,0

    idx = 0
    cntsSorted = sorted(cnts, key=lambda contours: cv2.contourArea(contours))
    image_candidates = []
    stripidx=0
    color_checkeridx=0
    # loop through each of the contours greater than Area/lower_Ratio.. The first contour would be color_checker and the second one is strip
    for c in cntsSorted:
        if cv2.contourArea(c) > (Area / lower_Ratio):


            idx=idx+1
            x, y, w, h = cv2.boundingRect(c)
            if (h/w)>3 or (w/h)>3 :
                x, y, w, h = cv2.boundingRect(c)
                Strip = image[y:y + h, x:x + w]
                image_candidates.append(Strip)
                # cv2.imwrite("Strip.jpg", Strip)
                print('Strip', cv2.contourArea(c))
                stripidx = idx-1

            else:
                rect = cv2.minAreaRect(c)
                color_checker = crop_minAreaRect(image, rect)
                image_candidates.append(color_checker)
                # print('this is shape',np.shape(color_checker)[0]*np.shape(color_checker)[1])
                print('colorcheckerarea',cv2.contourArea(c))
                # cv2.imwrite("colorchecker.jpg", color_checker)
                color_checkeridx=idx-1
                #show_Image(color_checker, "colorchecker"


    return image_candidates,stripidx,color_checkeridx