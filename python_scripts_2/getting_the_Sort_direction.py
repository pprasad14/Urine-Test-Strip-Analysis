def get_sort_direction(X,Y,siemen_x,siemens_y):

    if X>Y:
        centroid = X/2
        if centroid>siemen_x :
            sort="right-to-left"

        else :
            sort="left-to-right"

    else:
        centroid=Y/2
        if centroid > siemens_y:
         sort="bottom-to-top"

        else :
         sort="top-to-bottom"


    return sort
