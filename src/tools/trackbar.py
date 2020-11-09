import cv2
import numpy as np


def nothing(x):
    pass


cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)

cv2.createTrackbar("HH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("HS", "Tracking", 255, 255, nothing)
cv2.createTrackbar("HV", "Tracking", 255, 255, nothing)

while True:
    frame =cv2.imread("/home/boniface/catkin_ws/src/ivr_assignment/src/images/target_only/c1_bottom_left.png")
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    l_h=cv2.getTrackbarPos("LH","Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    h_h = cv2.getTrackbarPos("HH", "Tracking")
    h_s = cv2.getTrackbarPos("HS", "Tracking")
    h_v = cv2.getTrackbarPos("HV", "Tracking")


    l_r=np.array([l_h,l_s,l_v])
    u_r=np.array([h_h,h_s,h_v])

    mask=cv2.inRange(hsv,l_r,u_r)
    res=cv2.bitwise_and(frame,frame,mask=mask)

    cv2.imshow("Frame",frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", res)


    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord("s"):
        print(frame.shape)
        cv2.destroyAllWindows()
