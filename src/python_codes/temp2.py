import cv2
import numpy as np
import statistics
import imutils
import os
import matplotlib.pyplot as plt

arr = os.listdir("/home/boniface/catkin_ws/src/ivr_assignment/src/images/target_only")


#img = cv2.imread("/home/boniface/catkin_ws/src/ivr_assignment/src/images/target_only/[239].png")
template = cv2.imread("/home/boniface/catkin_ws/src/ivr_assignment/src/images/cropped.png",0)
#template = cv2.Canny(template, 50, 200)

#mask1 = cv2.Canny(mask1, 50, 200)

threshold = 0.9
w, h = template.shape[::-1]

#-----------------------------


for i in range(len(arr)):
    img = cv2.imread("/home/boniface/catkin_ws/src/ivr_assignment/src/images/target_only/" + str(arr[i]))
    hsv_convert_1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_convert_1, (10, 202, 0), (27, 255, 255))

    startX = []
    startY = []
    endX = []
    endY = []


    for scale in np.linspace(0.79, 1.0, 5)[::-1]:


        found = None
        resized = imutils.resize(mask1, width = int(mask1.shape[1] * scale))
        r = mask1.shape[1]/float(resized.shape[1])
        if resized.shape[0] < h or resized.shape[1] < w:
            break

        res = cv2.matchTemplate(resized,template,cv2.TM_CCOEFF_NORMED)
        loc = np.where( res > threshold)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

        #(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        #(endX, endY) = (int((maxLoc[0] + w) * r), int((maxLoc[1] + h) * r))
        # draw a bounding box around the detected result and display the image


        startX.append(int(maxLoc[0] * r))
        startY.append(int(maxLoc[1] * r))
        endX.append(int((maxLoc[0] + w) * r))
        endY.append(int((maxLoc[1] + h) * r))

        #cv2.rectangle(img, (int(startX), int(startY)), (int(endX), int(endY)), (0, 0, 255), 2)

    centerX = (statistics.mean(endX) + statistics.mean(startX))/2
    centerY = (statistics.mean(endY) + statistics.mean(startY))/2
    img = cv2.circle(img, (int(centerX), int(centerY)), radius=2, color=(255, 255, 255), thickness=-1)
    cv2.imwrite("/home/boniface/catkin_ws/src/ivr_assignment/src/images/" +str(arr[i]), img)



    #centerX = (statistics.mean(startX) + statistics.mean(endX))/2
    #centerY = (statistics.mean(startY) + statistics.mean(endY))/2

    #print(centerX)
    #print(centerY)
        #cv2.imshow("Image", img)
        #cv2.waitKey(300)
        #cv2.destroyAllWindows()





#    key = cv2.waitKey(3000)#pauses for 3 seconds before fetching next image
#    if key == 27:#if ESC is pressed, exit loop
#        cv2.destroyAllWindows()
#        break


#------------------------------
#res = cv2.matchTemplate(mask1,template,cv2.TM_CCOEFF_NORMED)
#loc = np.where( res > threshold)

#for pt in zip(*loc[::-1]):
#    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 3)




#cv2.imshow("win2", img)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
