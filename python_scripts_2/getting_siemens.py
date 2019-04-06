import cv2
import show_image
from show_image import *
import numpy as np


def get_strip_position(strip):

    siemens_gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(siemens_gray, 90, 255, cv2.THRESH_BINARY_INV)
    siemens_canny = cv2.Canny(th, 10, 15)
    siemens_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    siemens_closed = cv2.morphologyEx(siemens_canny, cv2.MORPH_CLOSE, siemens_kernel, iterations=10)
    #show_Image(np.hstack([siemens_canny,siemens_closed,th]),'siemens_closed')
    (_, siemens_cnts, hie) = cv2.findContours(siemens_closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(siemens_cnts)==0 :
        return 0,0
    maxcant = max(siemens_cnts, key=lambda contours: cv2.contourArea(contours))
    siemens_x, siemens_y, siemens_w, siemens_h = cv2.boundingRect(maxcant)
    siemens_x = siemens_x + siemens_w
    siemens_y = siemens_y + siemens_h
    return siemens_x, siemens_y
