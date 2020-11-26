#!/usr/bin/env python

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


class image_converter:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize a publisher to send images from camera1 to a topic named image_topic1
        self.image_pub1 = rospy.Publisher("image_topic1", Image, queue_size=1)
        self.image_pub2 = rospy.Publisher("image_topic2", Image, queue_size=1)
        #Initialize  a publisher to send joints angular posiion toa topic called joints_pos
        self.joints_pub=rospy.Publisher("joints_pos",Float64MultiArray,queue_size=10)

        #initialize a publisher for the four angles
        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)


        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.callback1)
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)

        #intialize variables to store
        self.time_trajectory = rospy.get_time()
        self.red = np.array([0.0, 0.0, 0.0, 0.0], dtype='float64')
        self.green = np.array([0.0, 0.0, 0.0, 0.0], dtype='float64')
        self.p2m = np.array([0.0], dtype='float64')
        self.ja4 = np.array([0.0], dtype='float64')
        # initialize errors
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
        self.time_previous_step2 = np.array([rospy.get_time()], dtype='float64')
        # initialize error and derivative of error for trajectory tracking
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

    # Recieve data from camera 1, process it, and publish
    def callback1(self, data):
        # Recieve the image
        try:
            self.image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback2(self, data):
        # Recieve the image
        try:
            self.image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        #Blob detection-------------------------------------------------------

        def detect_red(self,image1, image2):
            #smooth the image and reduce noise
            image_gau_blur1 = cv2.GaussianBlur(image1, (1, 1), 0)
            #convert colours to HSV
            hsv1 = cv2.cvtColor(image_gau_blur1, cv2.COLOR_BGR2HSV)
            #set the HSV values for red
            lower_red1 = np.array([0, 200, 0])
            higher_red1 = np.array([0, 255, 255])
            #Apply threshold to seperate the blob from rest of the robot
            red_range1 = cv2.inRange(hsv1, lower_red1, higher_red1)
            res_red1 = cv2.bitwise_and(image_gau_blur1, image_gau_blur1, mask=red_range1)
            #convert to grey scale
            red_s_gray1 = cv2.cvtColor(res_red1, cv2.COLOR_BGR2GRAY)
            #Detect the edges
            canny_edge1 = cv2.Canny(red_s_gray1, 30, 70)
            #Find the contours
            contours1, hierarchy1 = cv2.findContours(canny_edge1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #Find the center coordinates and the radius of the blob
            (x1, y1), radius1 = cv2.minEnclosingCircle(contours1[0])
            #convert to integers
            cy, cz1 = (int(x1), int(y1))
            radius1 = int(radius1)

            #similar to above, but for image 2
            image_gau_blur2 = cv2.GaussianBlur(image2, (1, 1), 0)
            hsv2 = cv2.cvtColor(image_gau_blur2, cv2.COLOR_BGR2HSV)
            lower_red2 = np.array([0, 200, 0])
            higher_red2 = np.array([0, 255, 255])
            red_range2 = cv2.inRange(hsv2, lower_red2, higher_red2)
            res_red2 = cv2.bitwise_and(image_gau_blur2, image_gau_blur2, mask=red_range2)
            red_s_gray2 = cv2.cvtColor(res_red2, cv2.COLOR_BGR2GRAY)
            canny_edge2 = cv2.Canny(red_s_gray2, 30, 70)
            contours2, hierarchy2 = cv2.findContours(canny_edge2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            (x2, y2), radius2 = cv2.minEnclosingCircle(contours2[0])
            cx, cz2 = (int(x2), int(y2))
            radius2 = int(radius2)

            return np.array([cx, cy, cz1, cz2])

        def detect_blue(self,image1, image2):
            #similar approach as detect_blue but different colour threshold
            image_gau_blur1 = cv2.GaussianBlur(image1, (1, 1), 0)
            hsv1 = cv2.cvtColor(image_gau_blur1, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([70, 0, 0])
            higher_red1 = np.array([255, 255, 255])
            red_range1 = cv2.inRange(hsv1, lower_red1, higher_red1)
            res_red1 = cv2.bitwise_and(image_gau_blur1, image_gau_blur1, mask=red_range1)
            red_s_gray1 = cv2.cvtColor(res_red1, cv2.COLOR_BGR2GRAY)
            canny_edge1 = cv2.Canny(red_s_gray1, 30, 70)
            contours1, hierarchy1 = cv2.findContours(canny_edge1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            (x1, y1), radius1 = cv2.minEnclosingCircle(contours1[0])
            cy, cz1 = (int(x1), int(y1))
            radius1 = int(radius1)


            image_gau_blur2 = cv2.GaussianBlur(image2, (1, 1), 0)
            hsv2 = cv2.cvtColor(image_gau_blur2, cv2.COLOR_BGR2HSV)
            lower_red2 = np.array([70, 0, 0])
            higher_red2 = np.array([255, 255, 255])
            red_range2 = cv2.inRange(hsv2, lower_red2, higher_red2)
            res_red2 = cv2.bitwise_and(image_gau_blur2, image_gau_blur2, mask=red_range2)
            red_s_gray2 = cv2.cvtColor(res_red2, cv2.COLOR_BGR2GRAY)
            canny_edge2 = cv2.Canny(red_s_gray2, 30, 70)
            contours2, hierarchy2 = cv2.findContours(canny_edge2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            (x2, y2), radius2 = cv2.minEnclosingCircle(contours2[0])
            cx, cz2 = (int(x2), int(y2))
            radius2 = int(radius2)

            return np.array([cx, cy, cz1, cz2])

        def detect_green(self,image1, image2):
            #similar approach as detect_blue but different colour threshold
            image_gau_blur1 = cv2.GaussianBlur(image1, (1, 1), 0)
            hsv1 = cv2.cvtColor(image_gau_blur1, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([55, 0, 0])
            higher_red1 = np.array([100, 255, 255])
            red_range1 = cv2.inRange(hsv1, lower_red1, higher_red1)
            res_red1 = cv2.bitwise_and(image_gau_blur1, image_gau_blur1, mask=red_range1)
            red_s_gray1 = cv2.cvtColor(res_red1, cv2.COLOR_BGR2GRAY)
            canny_edge1 = cv2.Canny(red_s_gray1, 30, 70)
            contours1, hierarchy1 = cv2.findContours(canny_edge1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            (x1, y1), radius1 = cv2.minEnclosingCircle(contours1[0])
            cy, cz1 = (int(x1), int(y1))
            radius1 = int(radius1)


            image_gau_blur2 = cv2.GaussianBlur(image2, (1, 1), 0)
            hsv2 = cv2.cvtColor(image_gau_blur2, cv2.COLOR_BGR2HSV)
            lower_red2 = np.array([55, 0, 0])
            higher_red2 = np.array([100, 255, 255])
            red_range2 = cv2.inRange(hsv2, lower_red2, higher_red2)
            res_red2 = cv2.bitwise_and(image_gau_blur2, image_gau_blur2, mask=red_range2)
            red_s_gray2 = cv2.cvtColor(res_red2, cv2.COLOR_BGR2GRAY)
            canny_edge2 = cv2.Canny(red_s_gray2, 30, 70)
            contours2, hierarchy2 = cv2.findContours(canny_edge2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            (x2, y2), radius2 = cv2.minEnclosingCircle(contours2[0])
            cx, cz2 = (int(x2), int(y2))
            radius2 = int(radius2)

            return np.array([cx, cy, cz1, cz2])

        def detect_yellow(self,image1, image2):
            #similar approach as detect_blue but different colour threshold
            image_gau_blur1 = cv2.GaussianBlur(image1, (1, 1), 0)
            hsv1 = cv2.cvtColor(image_gau_blur1, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([16, 244, 0])
            higher_red1 = np.array([51, 255, 255])
            red_range1 = cv2.inRange(hsv1, lower_red1, higher_red1)
            res_red1 = cv2.bitwise_and(image_gau_blur1, image_gau_blur1, mask=red_range1)
            red_s_gray1 = cv2.cvtColor(res_red1, cv2.COLOR_BGR2GRAY)
            canny_edge1 = cv2.Canny(red_s_gray1, 30, 70)
            contours1, hierarchy1 = cv2.findContours(canny_edge1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            (x1, y1), radius1 = cv2.minEnclosingCircle(contours1[0])
            cy, cz1 = (int(x1), int(y1))
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

            return np.array([cx, cy, cz1, cz2])

        def detect_blue_contours(image1):
            #similar to detect_red(), this one only returns the positions of the contour
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
            #similar to detect_red(), this one only returns the positions of the contour
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

            return np.array([contours1])

        def get_y1_y2(yellow_contours, blue_contours):
            #finds the z value at the top of yellow blob
            y1 = np.min(yellow_contours, axis = 0)
            y1 = y1[0][0]
            y1 = y1[:,1]

            #finds the z value at the bottom of blue blob
            y2 = np.max(blue_contours, axis = 0)
            y2 = y2[0][0]
            y2 = y2[:,1]

            return y1, y2

        def pixelTometer(self, image1,image2):
            #gets the contour coordinates of the blue and yellow
            yellow_contours = detect_yellow_contours(image2)
            blue_contours = detect_blue_contours(image2)
            #finds the z value of center of mass of blue blob
            y2 = detect_blue(self, image1, image2)
            y2 = y2[3]

            #returns the position of arm 1 ends
            y1, y2 = get_y1_y2(yellow_contours, blue_contours)

            #get the pixel to meter ratio by dividing arm1 length by pixel distance calculated
            p2m = 2.5/(y1 - y2)
            #65 is the best number

            return p2m

        #----------------------------------------------------------------------------------------------
        #Angle Detection starts here
        def detect_angles_blob(self,image1,image2):
            #Calculate the pixel to meter ratio
            try:
                p=pixelTometer(self,image1,image2)
                self.p2m = p
            except Exception as e:
                p = self.p2m

            #find the positions of the blob

            try:
                green = detect_green(self, image1, image2)
                self.green = green
            except Exception as e:
                green = self.green

            try:
                red = detect_red(self, image1, image2)
                self.red = red
            except Exception as e:
                red = self.red

            #convert to meters

            p=pixelTometer(self,image1,image2)
            yellow=detect_yellow(self,image1,image2)
            blue=detect_blue(self,image1,image2)

            #convert from pixel frame to camera frame on z value
            green[2] = 800 - green[2]
            yellow[2] = 800 - yellow[2]
            red[2] = 800 - red[2]

            #get ja1, ja and ja3
            ja1=0.0
            ja3 = get_ja3(green, yellow, p)
            ja2 = get_ja2(green, yellow, p, ja3)


            try:
                green = detect_green(self, image1, image2)
                self.green = green
            except Exception as e:
                green = self.green

            try:
                red = detect_red(self, image1, image2)
                self.red = red
            except Exception as e:
                red = self.red

            yellow=p*detect_yellow(self,image1,image2)
            blue=p*detect_blue(self,image1,image2)

            #get ja4 value
            ja4 = np.arctan2((green[2] - red[2]), -(green[1] - red[1]))-np.pi/2-ja2

            return np.array([ja1,ja2,ja3,ja4])

        def angle_trajectory(self):
            #the angle coordinates given to the target
            curr_time = np.array([rospy.get_time() - self.time_trajectory])
            ja1 = 0.1
            ja2 = float((np.pi / 2) * np.sin((np.pi / 15) * curr_time))
            ja3 = float((np.pi / 2) * np.sin((np.pi / 18) * curr_time))
            ja4 = float((np.pi / 2) * np.sin((np.pi / 20) * curr_time))
            return np.array([ja1, ja2, ja3, ja4])

        def get_ja3(green_posn, yellow_posn, p):
            #find the distance between green and yellow
            green = green_posn - yellow_posn
            #convert the distance to meter
            X_green = green[0]*p
            #X_green[0] cannot be greater than 3.5 or less than -3.5.
            #if the code reads that, it might be a pixel error. Therefore we are forcing the system to assume its max value
            if X_green > 3.5:
                X_green = 3.5
            elif X_green < -3.5:
                X_green = -3.5

            ja3 = np.arcsin(X_green/ 3.5)
            return ja3

        def get_ja2(green_posn, yellow_posn, p, ja3):
            green = green_posn - yellow_posn
            Y_green = green[1]*p

            #Y_green[0] cannot be greater than 3.5 or less than -3.5.
            #if the code reads that, it might be a pixel error. Therefore we are forcing the system to assume its max value
            if Y_green[0] > 3.5:
                Y_green[0] = 3.5
            elif Y_green[0] < -3.5:
                Y_green[0] = -3.5

            #calculate the value before being supplied into arcsin()
            arc_sin_val = np.round(Y_green[0]/(-3.5*np.cos(ja3)),2)

            #value inside the arcsin() cannot be greater than 1 or less than -1
            #if the number is greater or lower, we are focing it to accept the largest possible value
            if arc_sin_val > 1:
                arc_sin_val = 1
            elif arc_sin_val < -1:
                arc_sin_val = -1

            ja2 = np.arcsin(arc_sin_val)

            return ja2

        self.joints = Float64MultiArray()
        #get the joint angles from computer vision
        self.joints.data = detect_angles_blob(self,self.image1,self.image2)
        #get the joint angles generated automatically
        ja1,ja2,ja3,ja4=angle_trajectory(self)
        self.joint1 = Float64()
        self.joint1.data = 0
        self.joint2 = Float64()
        self.joint2.data = ja2
        self.joint3 = Float64()
        self.joint3.data = ja3
        self.joint4 = Float64()
        self.joint4.data = ja4




        #print(curr_time)

        try:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.image1, "bgr8"))
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.image2, "bgr8"))
            self.joints_pub.publish(self.joints)
            self.robot_joint1_pub.publish(self.joint1)
            self.robot_joint2_pub.publish(self.joint2)
            self.robot_joint3_pub.publish(self.joint3)
            self.robot_joint4_pub.publish(self.joint4)


        except CvBridgeError as e:
            print(e)
# call the class
def main(args):
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
