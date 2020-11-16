import numpy as np
import cv2



def detect_blue_contours(image1):
    image_gau_blur1 = cv2.GaussianBlur(image1, (1, 1), 0)
    hsv1 = cv2.cvtColor(image_gau_blur1, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([70, 0, 0])
    higher_red1 = np.array([255, 255, 255])
    red_range1 = cv2.inRange(hsv1, lower_red1, higher_red1)
    res_red1 = cv2.bitwise_and(image_gau_blur1, image_gau_blur1, mask=red_range1)
    red_s_gray1 = cv2.cvtColor(res_red1, cv2.COLOR_BGR2GRAY)
    canny_edge1 = cv2.Canny(red_s_gray1, 30, 70)
    contours1, hierarchy1 = cv2.findContours(canny_edge1,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return np.array([contours1])

def detect_yellow_contours(image1):
    image_gau_blur1 = cv2.GaussianBlur(image1, (1, 1), 0)
    hsv1 = cv2.cvtColor(image_gau_blur1, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([16, 244, 0])
    higher_red1 = np.array([51, 255, 255])
    red_range1 = cv2.inRange(hsv1, lower_red1, higher_red1)
    res_red1 = cv2.bitwise_and(image_gau_blur1, image_gau_blur1, mask=red_range1)
    red_s_gray1 = cv2.cvtColor(res_red1, cv2.COLOR_BGR2GRAY)
    canny_edge1 = cv2.Canny(red_s_gray1, 30, 70)
    contours1, hierarchy1 = cv2.findContours(canny_edge1,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return np.array([contours1])


def detect_yellow(image1,image2):
    image_gau_blur1 = cv2.GaussianBlur(image1, (1, 1), 0)
    hsv1 = cv2.cvtColor(image_gau_blur1, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([16, 244, 0])
    higher_red1 = np.array([51, 255, 255])
    red_range1 = cv2.inRange(hsv1, lower_red1, higher_red1)
    res_red1 = cv2.bitwise_and(image_gau_blur1, image_gau_blur1, mask=red_range1)
    red_s_gray1 = cv2.cvtColor(res_red1, cv2.COLOR_BGR2GRAY)
    canny_edge1 = cv2.Canny(red_s_gray1, 30, 70)
    contours1, hierarchy1 = cv2.findContours(canny_edge1,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    (x1, y1), radius1 = cv2.minEnclosingCircle(contours1[0])
    cy,cz1 = (int(x1), int(y1))
    radius1 = int(radius1)



    image_gau_blur2 = cv2.GaussianBlur(image2, (1, 1), 0)
    hsv2 = cv2.cvtColor(image_gau_blur2, cv2.COLOR_BGR2HSV)
    lower_red2 = np.array([16, 244, 0])
    higher_red2 = np.array([51, 255, 255])
    red_range2 = cv2.inRange(hsv2, lower_red2, higher_red2)
    res_red2 = cv2.bitwise_and(image_gau_blur2, image_gau_blur2, mask=red_range2)
    red_s_gray2 = cv2.cvtColor(res_red2, cv2.COLOR_BGR2GRAY)
    canny_edge2 = cv2.Canny(red_s_gray2, 30, 70)
    contours2, hierarchy2 = cv2.findContours(canny_edge2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    (x2, y2), radius2 = cv2.minEnclosingCircle(contours2[0])
    cx, cz2 = (int(x2), int(y2))
    radius2 = int(radius2)

    return np.array([cx,cy,cz1,cz2])

def get_y1_y2(yellow_contours, blue_contours):

    y2 = np.max(np.array(blue_contours), axis = 0)
    y2 = y2[0][1]
    y2 = y2[:,1]

    y1 = np.min(yellow_contours, axis = 0)
    y1 = y1[0][1]
    y1 = y1[:,1]

    return y1, y2


def pixelTometer(image1):
    yellow_contours = detect_yellow_contours(image)
    blue_contours = detect_blue_contours(image)
    y1, y2 = get_y1_y2(yellow_contours, blue_contours)

    p2m = 2.5/(y1 - y2)

    return p2m




image1 = cv2.imread("/home/boniface/catkin_ws/src/ivr_assignment/src/tools/image_copy1.png")
image2 = cv2.imread("/home/boniface/catkin_ws/src/ivr_assignment/src/tools/image_copy2.png")

d = detect_yellow(image1,image2)

d[0]

yellow_contours =detect_yellow_contours(image1)

blue_contours = detect_blue_contours(image1)

y1, y2 = get_y1_y2(yellow_contours, blue_contours)

x1, y1 = yellow_y, int(y1)
x2, y2 = yellow_y, int(y2)

a = np.max(np.array(blue_contours), axis = 0)

a[0][1]

line_thickness = 2
cv2.line(image1, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)

cv2.imwrite('img_blue.png', image1)
