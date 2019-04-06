import cv2


class CustomContour(object):

    def __init__(self, index, contour, heirarchy):

        self.index = index
        self.contour = contour
        self.heirarchy = heirarchy
        peri = cv2.arcLength(contour, True)
        self.approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
        self.area = cv2.contourArea(contour)
        self.corners = len(self.approx)
        self.width = None


        # compute the center of the contour
        M = cv2.moments(contour)

        if M["m00"]:
            self.cX = int(M["m10"] / M["m00"])
            self.cY = int(M["m01"] / M["m00"])

            #if self.cX == 188 and self.cY == 93:
            #    log.warning("CustomContour M %s" % pformat(M))
        else:
            self.cX = None
            self.cY = None

    def __str__(self):
        return "Contour #%d (%s, %s)" % (self.index, self.cX, self.cY)

def sort_by_row_col(contours, row,column):
    """
    Given a list of contours sort them starting from the upper left corner
    and ending at the bottom right corner
    """
    result = []
    num_squares = len(contours)

    for row_index in range(row):

        # We want the 'size' squares that are closest to the top
        tmp = []
        for con in contours:
            tmp.append((con.cY, con.cX))
        top_row = sorted(tmp)[:column]

        # Now that we have those, sort them from left to right
        top_row_left_right = []
        for (cY, cX) in top_row:
            top_row_left_right.append((cX, cY))
        top_row_left_right = sorted(top_row_left_right)


        contours_to_remove = []
        for (target_cX, target_cY) in top_row_left_right:
            for con in contours:

                if con in contours_to_remove:
                    continue

                if con.cX == target_cX and con.cY == target_cY:
                    result.append(con)
                    contours_to_remove.append(con)
                    break

        for con in contours_to_remove:
            contours.remove(con)

    return result
