import cv2
import numpy as np
import show_image
from show_image import *
import getting_siemens
from getting_siemens import *
import getting_the_Sort_direction
from getting_the_Sort_direction import *
import Sort_contours
from Sort_contours import *
def Strip_analysis (Strip,gb1,gb2,it1,it2,cn1,cn2):
    #Get the greyscale image and apply gaussian blur
    Strip_gray=cv2.cvtColor(Strip,cv2.COLOR_BGR2GRAY)
    Strip_blurred= cv2.GaussianBlur(Strip_gray,(gb1,gb2),0)

    #getting the edges of the image and closing the contours
    Strip_canny=cv2.Canny(Strip_blurred,cn1,cn2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    Strip_dilation = cv2.dilate(Strip_canny,kernel,iterations =it1)
    Strip_closed = cv2.morphologyEx(Strip_dilation, cv2.MORPH_CLOSE, kernel, iterations=it2)

    #draw the strip contours and do the further processing
    (_,Strip_cnts, _) = cv2.findContours(Strip_closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   # show_Image(np.hstack([Strip_canny,Strip_closed,Strip_dilation]),"output")

    #getting the lower and upper threshold ratio for the contours
    Strip_area=np.shape(Strip)[0]*np.shape(Strip)[1]

    lower_threshold=Strip_area/135
    upper_threshold=(Strip_area/10)
    # print("lower",lower_threshold,"upper",upper_threshold)

    #sorrt the contours based on the decided criteria
    Siemen_x, Siemens_y = get_strip_position(Strip)

    Y, X, Z = np.shape(Strip)
    Sort = get_sort_direction(X, Y, Siemen_x, Siemens_y)
    Strip_cnts, bounding_boxes = sort_contours(Strip_cnts, Sort)

    idx=0
    Strip_candidates=[]

    # print("length",len(Strip_cnts))
    for c2 in Strip_cnts:
        #print(cv2.contourArea(c2))

        if cv2.contourArea(c2)>(lower_threshold)and cv2.contourArea(c2)<(upper_threshold):
            Strip_x, Strip_y, Strip_w, Strip_h = cv2.boundingRect(c2)
            # print(Sort, upper_threshold, (Strip_y + Strip_h), Siemens_y)
            if Sort=='right-to-left' :
                if (Strip_x+Strip_w>Siemen_x) :
                    Strip_candidates.append(c2)

            elif Sort == 'left-to-right':
                if (Strip_x + Strip_w < Siemen_x):
                    Strip_candidates.append(c2)

            elif Sort == 'bottom-to-top':
                    if (Strip_y + Strip_h) > Siemens_y:
                        Strip_candidates.append(c2)


            elif Sort == 'top-to-bottom':
                    if (Strip_y + Strip_h) < Siemens_y:
                        Strip_candidates.append(c2)



    # print(len(Strip_candidates),'length')
    return Strip_candidates

