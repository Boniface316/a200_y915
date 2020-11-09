import cv2
import numpy as np
import statistics
import imutils
import os
import matplotlib.pyplot as plt

i = 3

arr = os.listdir("/home/boniface/catkin_ws/src/ivr_assignment/src/images/target_only")
img = cv2.imread("/home/boniface/catkin_ws/src/ivr_assignment/src/images/target_only/" + str(arr[i]))


template = cv2.imread("/home/boniface/catkin_ws/src/ivr_assignment/src/images/cropped.png",0)

threshold = 0.9
w, h = template.shape[::-1]

#-----------------------------





hsv_convert_1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv_convert_1, (10, 202, 0), (27, 255, 255))


for scale in np.linspace(0.79, 1.0, 5)[::-1]:
    print(scale)
    found = None
    resized = imutils.resize(mask1, width = int(mask1.shape[1] * scale))
    r = mask1.shape[1]/float(resized.shape[1])
    if resized.shape[0] < h or resized.shape[1] < w:
        break

    res = cv2.matchTemplate(resized,template,cv2.TM_CCOEFF_NORMED)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)

    if found is None or maxVal > found[0]:
        found = (maxVal, maxLoc, r)

    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + w) * r), int((maxLoc[1] + h) * r))
    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)

    #cv2.imwrite("/home/boniface/catkin_ws/src/ivr_assignment/src/images/" +str(arr[i]), img)
    #print(maxVal)
    cv2.imshow("Image", img)

    cv2.waitKey(1000)
    cv2.destroyAllWindows()
