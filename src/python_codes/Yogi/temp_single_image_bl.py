import roslib
import sys
import rospy
import cv2
import math
import imutils
import statistics
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial import distance as dist


#arr = os.listdir("/home/boniface/catkin_ws/src/ivr_assignment/src/images/target_only")

image1 = cv2.imread("/home/boniface/catkin_ws/src/ivr_assignment/src/images/black_robot/Cam 1/[309].png")

mask = cv2.inRange(image1, (0, 0, 0), (180, 255, 30))
thresh, dst = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY_INV )

base_template = cv2.imread("/home/boniface/catkin_ws/src/ivr_assignment/src/python_codes/Yogi/base_template.png", 0)

dst.shape
base_template.shape


res = cv2.matchTemplate(dst,base_template,cv2.TM_CCOEFF_NORMED)
(_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)

w, h = base_template.shape
startX = maxLoc[0]
startY = maxLoc[1]
endX = maxLoc[0] + w
endY = maxLoc[1] + h




maxVal
maxLoc

startX
startY

endX
endY
# Radius of circle

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

# Using cv2.circle() method
# Draw a circle with blue line borders of thickness of 2 px
image = cv2.circle(image1, (endX, endY), 2, color, thickness)

image = cv2.rectangle(image1, (startX, startY), (endY, endX), color, thickness)
cv2.imwrite("tempalte.png", image)


mask = np.zeros(image1.shape, dtype=np.uint8)
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray.png", gray)
thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY )[1]
kernel_erode = np.ones((3, 3), np.uint8)
kernel_dilate = np.ones((1, 1), np.uint8)
mask = cv2.erode(mask, kernel_erode, iterations=6)
cv2.imwrite("erode.png", mask)
mask = cv2.dilate(mask, kernel_dilate, iterations=0)
cv2.imwrite("dilate.png", mask)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
pening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
canny = cv2.Canny(mask, 100, 170)
cv2.imwrite("canny.png", canny)
#cimg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 20, param1=20, param2=20, minRadius=0,  maxRadius=0)
