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
        #initialize a publisher for the robot end effector

        #initialize a publisher for the four angles
        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
        #Initialize the publisher for t target

        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.callback1)
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)
        #initialize a publisher to send desired trajectory
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

        #Blob detection starts here-------------------------------------------------------

        def detect_red(self,image1, image2):
            image_gau_blur1 = cv2.GaussianBlur(image1, (1, 1), 0)
            hsv1 = cv2.cvtColor(image_gau_blur1, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 200, 0])
            higher_red1 = np.array([0, 255, 255])
            red_range1 = cv2.inRange(hsv1, lower_red1, higher_red1)
            res_red1 = cv2.bitwise_and(image_gau_blur1, image_gau_blur1, mask=red_range1)
            red_s_gray1 = cv2.cvtColor(res_red1, cv2.COLOR_BGR2GRAY)
            canny_edge1 = cv2.Canny(red_s_gray1, 30, 70)
            contours1, hierarchy1 = cv2.findContours(canny_edge1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # cv2.drawContours(image1, contours1, -1, (255, 255, 255), 1)
            (x1, y1), radius1 = cv2.minEnclosingCircle(contours1[0])
            cy, cz1 = (int(x1), int(y1))
            radius1 = int(radius1)
            # cv2.circle(image1, (cy, cz1), radius1, (255, 255, 255), 2)

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
            (x1, y1), radius1 = cv2.minEnclosingCircle(contours1[0])
            cy,cz1 = (int(x1), int(y1))

            return np.array([contours1])

        def get_y1_y2(yellow_contours, blue_contours):
            y1 = np.min(yellow_contours, axis = 0)
            y1 = y1[0][0]
            y1 = y1[:,1]

            y2 = np.max(blue_contours, axis = 0)
            y2 = y2[0][0]
            y2 = y2[:,1]

            return y1, y2

        def pixelTometer(self, image1,image2):

            yellow_contours = detect_yellow_contours(image2)
            blue_contours = detect_blue_contours(image2)
            y2 = detect_blue(self, image1, image2)
            y2 = y2[3]

            y1, y2 = get_y1_y2(yellow_contours, blue_contours)

            p2m = 2.5/(y1 - y2)
            #65 is the best number

            return p2m

        #----------------------------------------------------------------------------------------------
        #Angle Detection starts here
        def detect_angles_blob(self,image1,image2):
            try:
                p=pixelTometer(self,image1,image2)
                self.p2m = p
            except Exception as e:
                p = self.p2m


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

            p=pixelTometer(self,image1,image2)
            yellow=detect_yellow(self,image1,image2)
            blue=detect_blue(self,image1,image2)

            green[2] = 800 - green[2]
            yellow[2] = 800 - yellow[2]
            red[2] = 800 - red[2]

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
            ja4 = np.arctan2((green[2] - red[2]), -(green[1] - red[1]))-np.pi/2-ja2

            return np.array([ja1,ja2,ja3,ja4])

        def angle_trajectory(self):
            curr_time = np.array([rospy.get_time() - self.time_trajectory])
            ja1 = 0.1
            ja2 = float((np.pi / 2) * np.sin((np.pi / 15) * curr_time))
            ja3 = float((np.pi / 2) * np.sin((np.pi / 18) * curr_time))
            ja4 = float((np.pi / 2) * np.sin((np.pi / 20) * curr_time))
            return np.array([ja1, ja2, ja3, ja4])

        def get_ja3(green_posn, yellow_posn, p):
            green = green_posn - yellow_posn
            X_green = green[0]*p
            if X_green > 3.5:
                X_green = 3.5
            elif X_green < -3.5:
                X_green = -3.5

            ja3 = np.arcsin(X_green/ 3.5)
            return ja3

        def get_ja2(green_posn, yellow_posn, p, ja3):
            green = green_posn - yellow_posn
            Y_green = green[1]*p

            if Y_green[0] > 3.5:
                Y_green[0] = 3.5
            elif Y_green[0] < -3.5:
                Y_green[0] = -3.5

            arc_sin_val = np.round(Y_green[0]/(-3.5*np.cos(ja3)),2)

            if arc_sin_val > 1:
                arc_sin_val = 1
            elif arc_sin_val < -1:
                arc_sin_val = -1

            ja2 = np.arcsin(arc_sin_val)

            return ja2

        self.joints = Float64MultiArray()
        self.joints.data = detect_angles_blob(self,self.image1,self.image2)
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
