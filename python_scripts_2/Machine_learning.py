from flask import Flask, render_template, request, jsonify
import os

import cv2
import numpy as np
import matplotlib as plot
import pandas as pd
import copy
from copy import deepcopy
import os
import sys
import custom_contour_py
from custom_contour_py import *
import Sort_contours
from Sort_contours import *
import show_image
from custom_contour_py import CustomContour
from show_image import *
import get_color_checker_contours
from get_color_checker_contours import *
import getting_siemens
from getting_siemens import *
import Get_Bgr_Get_Gr_ratio
from Get_Bgr_Get_Gr_ratio import *
import get_Cropped_image
from get_Cropped_image import *
import getting_the_Sort_direction
from getting_the_Sort_direction import *
import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
# import statsmodels.api as sm
import json
import algorithm_1
from algorithm_1 import *
import logging as log
import strip_analysis
from strip_analysis import *
import get_the_objects
from get_the_objects import *

from PIL import Image  # uses pillow
from werkzeug.utils import secure_filename

app = Flask(__name__)

# UPLOAD_FOLDER = r'static'
UPLOAD_FOLDER = r'img_static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def analyse_image(image_path) :
    # print ("1")
   # image_path="IMG-20180814-WA0002.jpg";
    if not os.path.exists(image_path) :
        print("Image path does not exist")
        return "{\"Error\":\"path_does_not_exist\"}"
    image = cv2.imread(image_path)
    if image is None :
        return "{\"Error\":\"the file is not a image\"}"


    image_candidates,sidx,cidx=get_objects(image,13,13,20,25,20)
    if len(image_candidates)==0:
        return "{\"Error\":\"no hierarchy\"}"


    if len(image_candidates)!=2 :
       for i in range(1,4) :
           if i ==1:
               image_candidates,sidx,cidx= get_objects(image,13,13,20,25,10)
               if len(image_candidates)==2 :
                   break
           elif i==2   :
               image_candidates ,sidx,cidx= get_objects(image, 13, 13, 20, 25, 20)
               if len(image_candidates) == 2 :
                   break

           elif i == 3:
               image_candidates,sidx,cidx = get_objects(image, 13, 13, 20, 25, 65)
               if len(image_candidates) == 2:
                   break

    if len(image_candidates)!=2 :
        return "{\"Error\":\"objects not found\"}"

    Strip =image_candidates[sidx]
    color_checker=image_candidates[cidx]
   #show_Image(image_candidates[1],'image_candidates[1]')
    if Strip is None :
      return "{\"Error\":\"Strip not found\"}"
    #getting the position of 'Siemens' text in the image
    Siemen_x,Siemens_y = get_strip_position(Strip)
    if Siemen_x==0 and Siemens_y==0 :
        return "{\"Error\":\"no hierarchy in Strip\"}"


    Y,X,Z=np.shape(Strip)
    Sort= get_sort_direction(X,Y,Siemen_x,Siemens_y)
    Strip_candidates=Strip_analysis(Strip,23,23,20,35,5,12)
    if len(Strip_candidates)!=11:
        for i in range(1,14):
            print(i)
            if i==1:
                Strip_candidates = Strip_analysis(Strip, 23, 23, 20, 35, 5, 12)
                if len(Strip_candidates) == 11:
                    break;

            elif i == 2:
                Strip_candidates = Strip_analysis(Strip, 11, 11,6,12, 12, 15)
                if len(Strip_candidates) == 11:
                    break;
            if i==3 :
                Strip_candidates=Strip_analysis(Strip,21,21,5,4,7,9)
                if len(Strip_candidates) == 11:
                    break;
            elif i==4 :
                Strip_candidates=Strip_analysis(Strip,15,15,0,13,12,15)
                if len(Strip_candidates) == 11:
                    break;


            elif i == 5:
                #dont change 3159
                 Strip_candidates = Strip_analysis(Strip, 27, 27, 7, 29, 9, 11)
                 if len(Strip_candidates) == 11:
                     break;
            if i == 6:
                 Strip_candidates = Strip_analysis(Strip, 25,25,13,25, 5, 8)
                 if len(Strip_candidates) == 11:
                     break;
            if i == 7:
                Strip_candidates = Strip_analysis(Strip, 23, 23, 15, 25, 5, 12)
                if len(Strip_candidates) == 11:
                     break;

            if i== 8:
                Strip_candidates = Strip_analysis(Strip, 61, 61, 2, 25, 5, 10)
                if len(Strip_candidates) == 11:
                    break;
            if i == 9:
                # dont change 3159
                Strip_candidates = Strip_analysis(Strip, 61, 61, 2, 35, 9, 12)
                if len(Strip_candidates) == 11:
                    break;
            if i == 9:
                # dont change 3159
                Strip_candidates = Strip_analysis(Strip, 61, 61, 2, 35, 9, 12)
                if len(Strip_candidates) == 11:
                    break;
            if i==10 :
                Strip_candidates = Strip_analysis(Strip, 43, 43, 3, 25, 5, 15)
                if len(Strip_candidates) == 11:
                    break;
            if i==11:
                Strip_candidates = Strip_analysis(Strip, 31, 31, 2, 25, 5, 7)
                if len(Strip_candidates) == 11:
                    break;
            if i==12:
                Strip_candidates = Strip_analysis(Strip, 31, 31, 2, 18, 5, 7)
                if len(Strip_candidates) == 11:
                    break;
                   #works for latest
            if i==13 :
                Strip_candidates = Strip_analysis(Strip, 31, 31, 5, 25, 10, 12)
                if len(Strip_candidates) == 11:
                    break;


    if len(Strip_candidates) !=11 and len(Strip_candidates) !=11 :
        return "{\"Error\":\"Strip_segmentation failed\"}"


    #sorrt the contours based on the decided criteria
    Strip_cnts,bounding_boxes= sort_contours(Strip_candidates,Sort)

    idx=0
    pad_dict={}
    padlist=["LEU","NIT","URO","PRO","PH","BLO","SG","KET","BIL","GLU","k"]
    # print("length",len(Strip_cnts))
    for c2 in Strip_cnts:

            Strip_x, Strip_y, Strip_w, Strip_h = cv2.boundingRect(c2)
            pad = Strip[Strip_y:Strip_y + Strip_h, Strip_x:Strip_x + Strip_w]
            M = cv2.moments(c2)

            cx = int((Strip_x+Strip_x+Strip_w)/2)
            cy = int((Strip_y+Strip_y+Strip_h)/2)




            idx+=1
            if idx<12:
                bgr = Strip[cy, cx]
                [mean_red, mean_green, mean_blue] = get_BGR(Strip, cx, cy)
                pad_dict[padlist[idx-1]]= [mean_red, mean_green,mean_blue]

                #show_Image(pad, "pad" + str(idx))


    #print(pad_dict)

    colorchecker=color_checker
    gray=cv2.cvtColor(colorchecker,cv2.COLOR_BGR2GRAY)

    cnts,candidates=get_the_contours(colorchecker,gray,7,7,13,13,0,20,8,10)

    if len(candidates)!=30 :
        for i in range(1,9):
            print (i)
            if i==1 :
                cnts, candidates = get_the_contours(colorchecker,gray, 5, 5, 11,11,0,20,8,10)
                if len(candidates) == 30 :
                    # print("breaking here")
                     break
            elif i==2 :
                cnts, candidates = get_the_contours(colorchecker,gray, 9, 9, 15, 15,4,20,8,10)
                # dont change.. for 3159
                if len(candidates) == 30:
                    break
            elif i==3:
                cnts, candidates = get_the_contours(colorchecker, gray, 7, 7, 21, 21, 9, 20, 5, 8)
                if len(candidates) == 30:
                    break
            elif i == 4:
                cnts, candidates = get_the_contours(colorchecker, gray, 7, 7, 13, 13, 4, 20, 8, 10)
                if len(candidates) == 30:
                     break
            #
            elif i == 5:
                cnts, candidates = get_the_contours(colorchecker, gray, 7, 7, 21, 21, 5, 16, 5, 8)
                if len(candidates) == 30:
                #dont change.. for 3159
                  break
            if i == 6:
                cnts, candidates = get_the_contours(colorchecker, gray, 6,6,17,17,5,25, 5,7)
                if len(candidates) == 30:
                    break
            # elif i == 7:
            #     cnts, candidates = get_the_contours(colorchecker, gray, 9,9, 15, 15, 10,10, 5,8)
            #     if len(candidates) == 30:
            #         break

    if len(candidates) != 30:
        return "{\"Error\":\"color_checker segmentation failed\"}"


    Maxcant = max(cnts, key=lambda cnts: cv2.contourArea(cnts))
    x, y, w, h = cv2.boundingRect(Maxcant)



    colorlist=[]
    if(w>h) :

       result = sort_by_row_col(deepcopy(candidates), 5, 6)
       if len(result) == 30: #else should be handled from here
               X25_Ratio = get_Red_Green_Ratio(colorchecker, result[24].cX, result[24].cY)
               X6_Ratio = get_Red_Green_Ratio(colorchecker, result[5].cX, result[5].cY)
               if X25_Ratio>2:
                   colorlist=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
                   # print("x25_yes")
               elif X6_Ratio>2 :
                   colorlist=[30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
                   # print("x6_yes")

    elif(h>w) :
       result = sort_by_row_col(deepcopy(candidates), 6, 5)
       if len(result) == 30:
           X1_Ratio = get_Red_Green_Ratio(colorchecker, result[0].cX, result[0].cY)
           X30_Ratio = get_Red_Green_Ratio(colorchecker, result[29].cX, result[29].cY)


           if X1_Ratio>2:
               colorlist=[25,19,13,7,1,26,20,14,8,2,27,21,15,9,3,28,22,16,10,4,29,23,17,11,5,30,24,18,12,6]
               # print("x1_yes")
           elif X30_Ratio>2 :
               colorlist =  [6, 12, 18, 24, 30, 5, 11, 17, 23, 29, 4, 10, 16, 22, 28, 3, 9, 15, 21, 27, 2, 8, 14, 20, 26, 1, 7, 13, 19, 25]
               # print("x29_yes")



    idx=0
    color_dict={}

    if len(colorlist)!=30 :
        return                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 "{\"Error\":\"color_checker segmentation failed\"}"

    if len(result)==30 :
        for con in result :
            if len(colorlist)==30 :
                color_dict[colorlist[idx]]=get_BGR(colorchecker,con.cX,con.cY)
            x, y, w, h = cv2.boundingRect(con.contour)
            idx += 1


    print (color_dict)

    pad= pd.DataFrame.from_dict(pad_dict, orient='index',
                           columns=['Mean_red', 'Mean_Green','Mean_Blue'])
    print (pad)
    pad.iloc[:,0]=pad.iloc[:,0].div(int(pad.loc['k','Mean_red']))
    pad.iloc[:,1]=pad.iloc[:,1].div(int(pad.loc['k','Mean_Green']))
    pad.iloc[:,2]=pad.iloc[:,2].div(int(pad.loc['k','Mean_Blue']))




    color_checker_CC= pd.DataFrame.from_dict(color_dict, orient='index',
                           columns=['Mean_red', 'Mean_Green','Mean_Blue'])
    color_checker_CC.iloc[:,0]=color_checker_CC.iloc[:,0].div(int(color_checker_CC.iloc[7,0]))
    color_checker_CC.iloc[:,1]=color_checker_CC.iloc[:,1].div(int(color_checker_CC.iloc[7,1]))
    color_checker_CC.iloc[:,2]=color_checker_CC.iloc[:,2].div(int(color_checker_CC.iloc[7,2]))


    color_checker_CS=pd.read_csv('ColorChecker30.csv',header=None,skiprows=2, names=['CC_no','Mean_red','Mean_Green','Mean_Blue'],index_col='CC_no')
    color_checker_CS.iloc[:,0]=color_checker_CS.iloc[:,0].div(int(color_checker_CS.iloc[7,0]))
    color_checker_CS.iloc[:,1]=color_checker_CS.iloc[:,1].div(int(color_checker_CS.iloc[7,1]))
    color_checker_CS.iloc[:,2]=color_checker_CS.iloc[:,2].div(int(color_checker_CS.iloc[7,2]))


    diff=color_checker_CC.sub(color_checker_CS)

    corrected_dict={}
    look_up_dict={}
    look_up_dict={'LEU':{'Neg':0.9079,  'Trace':0.82,'Small': 0.6627,'Moderate':0.4995,'Large':0.348},
    'NIT':{'Negative':0.9793,'Positive':0.8947},'URO':{0.2:0.8123,1:0.7053,2:0.626,4:0.5384,'> 8.0':0.4817},'PRO':{'Neg':0.8871,'Trace':0.7117,30:0.6096,
           100:0.5176,'>=300':0.4228},'BLO':{'Neg':0.6531,'Trace-Lysed':0.5031,'Moderate':0.2289},'SG':{'<=1.00':0.2173,'1.01':0.2781,'1.015':0.4088,'1.02':0.4522,'>=1.030':0.6096},'KET':
              {'Neg':0.7519,'Trace':0.6063,15:0.5122,40:0.3934,80:0.2534,'>=160':0.1685},'BIL':{'Neg':0.9404,'Small':0.8072,'Moderate':0.7263,
            'Large':0.6186},'GLU':{'Neg':0.7659,100:0.7346,250:0.6903,500:0.3729,'>=1000':0.3118},'PH':{'5.0':0.8893,5.5:0.8792,6.0:0.8336,6.5:0.768,7.0:0.561,7.5:0.4238,8.5:0.3273}}

    get__values(corrected_dict,"GLU",'Mean_Green',29,26,1,diff,color_checker_CC,pad,look_up_dict)
    get__values(corrected_dict,"BIL",'Mean_Green',28,1,19,diff,color_checker_CC,pad,look_up_dict)
    get__values(corrected_dict,"KET",'Mean_Green',2,24,1,diff,color_checker_CC,pad,look_up_dict)
    get__values(corrected_dict,"SG",'Mean_Green',26,29,28,diff,color_checker_CC,pad,look_up_dict)
    get__values(corrected_dict,"BLO",'Mean_Green',28,19,29,diff,color_checker_CC,pad,look_up_dict)
    get__values(corrected_dict,"PH",'Mean_red',12,28,26,diff,color_checker_CC,pad,look_up_dict)
    get__values(corrected_dict,"PRO",'Mean_red',29,26,6,diff,color_checker_CC,pad,look_up_dict)
    get__values(corrected_dict,"URO",'Mean_Green',2,25,24,diff,color_checker_CC,pad,look_up_dict)
    get__values(corrected_dict,"NIT",'Mean_Green',2,28,12,diff,color_checker_CC,pad,look_up_dict)
    get__values(corrected_dict,"LEU",'Mean_red',1,30,4,diff,color_checker_CC,pad,look_up_dict)


    json_string=json.dumps(corrected_dict)

    return json_string

    
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# def allowed_file(filename):
#     return '.' in filename and filename.split('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods = ['POST'])
def show():
    return jsonify({"test":"abc"})
	# return render_template('Machine_learning.html');
	# pass

@app.route('/upload', methods = ['POST'])
def upload():
    try:

        image = request.files['img']

        image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))

        # return jsonify({"status":"error"})

        # image_secure_name = secure_filename(image.filename)
        # image.save(os.path.join(app.config['UPLOAD_FOLDER'],image_secure_name))        
        
        im = Image.open(image)
        print ("Image dimension: {}".format(im.size))

        result = analyse_image(UPLOAD_FOLDER+'/'+image.filename)
        return result
        
        # result = analyse_image(UPLOAD_FOLDER+'/'+image_secure_name)
    
    except:
		# return jsonify({'error':'upload error'})
        return ({'error':'upload error'})
		
if __name__ == "__main__":
    app.run()