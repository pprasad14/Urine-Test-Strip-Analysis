import cv2

def show_Image(Image,Windowname) :
    cv2.namedWindow(Windowname,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(Windowname,1200,1200)
    cv2.imshow(Windowname,Image)
    cv2.waitKey(0)
    cv2.destroyWindow(Windowname)
