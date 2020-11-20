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
        self.vision_end_effector_pub=rospy.Publisher("vision_end_effector",Float64MultiArray,queue_size=10)
        self.fk_end_effector_pub = rospy.Publisher("fk_end_effector", Float64MultiArray, queue_size=10)

        self.actual_target_trajectory_pub = rospy.Publisher("actual_target_trajectory", Float64MultiArray,queue_size=10)
        self.vision_target_trajectory_pub = rospy.Publisher("vision_target_trajectory", Float64MultiArray,queue_size=10)
        #initialize a publisher for the four angles
        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
        #Initialize the publisher for t target
        self.target_x_pub = rospy.Publisher("/target/x_position_controller/command", Float64, queue_size=10)
        self.target_y_pub = rospy.Publisher("/target/y_position_controller/command", Float64, queue_size=10)
        self.target_z_pub = rospy.Publisher("/target/z_position_controller/command", Float64, queue_size=10)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.callback1)
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)
        #initialize a publisher to send desired trajectory
        self.time_trajectory = rospy.get_time()
        # initialize errors
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
        self.time_previous_step2 = np.array([rospy.get_time()], dtype='float64')
        # initialize error and derivative of error for trajectory tracking
        self.error = np.array([0.0, 0.0,0.0], dtype='float64')
        self.error_d = np.array([0.0, 0.0,0.0], dtype='float64')
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
            (x1, y1), radius1 = cv2.minEnclosingCircle(contours1[0])
            cy, cz1 = (int(x1), int(y1))
            radius1 = int(radius1)

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
            y1 = y1[0][1]
            y1 = y1[:,1]

            y2 = np.max(blue_contours, axis = 0)
            y2 = y2[0][1]
            y2 = y2[:,1]

            return y1, y2

        def pixelTometer(image1,image2):



            yellow_contours = detect_yellow_contours(image2)
            blue_contours = detect_blue_contours(image2)
            y2 = detect_blue(self, image1, image2)
            y2 = y2[3]

            y1, y2 = get_y1_y2(yellow_contours, blue_contours)

            p2m = 2.5/(y1 - y2)
            #65 is the best number


            return p2m


        #FK starts here--------------------------------------------------------------------------------
        def get_link_matrix(theta, d, a, alpha):
            Rz = np.matrix([[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            TransZ = np.matrix([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1.0, d], [0, 0, 0, 1.0]])
            TransX = np.matrix([[1.0, 0, 0, a], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
            Rx = np.matrix([[1, 0, 0, 0], [0, np.cos(alpha), -np.sin(alpha), 0], [0, np.sin(alpha), np.cos(alpha), 0], [0, 0, 0, 1]])
            link_matrix = Rz.dot(TransZ).dot(TransX).dot(Rx)
            return(link_matrix)

        def get_transormation_matrix(self, ja1, ja2, ja3, ja4):
            TC = get_link_matrix(np.pi/2, 0, 0, 0)
            T1 = get_link_matrix(ja1, 2.5, 0, np.pi/2)
            T2 = get_link_matrix(np.pi/2, 0, 0, 0)
            T3 = get_link_matrix(ja2, 0, 0, np.pi/2)
            T4 = get_link_matrix(ja3, 0, 3.5, -np.pi/2)
            T5 = get_link_matrix(ja4, 0, 3, 0)
            T0 = TC.dot(T1).dot(T2).dot(T3).dot(T4).dot(T5)
            T0 = np.round(T0, 2)

            return T0

        def get_EE_by_FK(self,ja1, ja2, ja3, ja4):
            T0 = get_transormation_matrix(self, ja1, ja2, ja3, ja4)
            x_d = T0[:3,3]

            return x_d

        def end_effector_position(self, image1, image2):
            try:
                p=pixelTometer(self,image1,image2)
                self.p2m = p
            except Exception as e:
                p = self.p2m
            yellow_posn = detect_yellow(self,image1, image2)
            red_posn = detect_red(self, image1, image2)
            yellow_posn[3] = 800 - yellow_posn[3]
            red_posn[3] = 800 - red_posn[3]

            cx, cy, cz1, cz2 = p * (red_posn - yellow_posn)
            ee_posn = np.array([cx, cy, cz2])
            ee_posn = np.round(ee_posn,1)
            return ee_posn

        def draw_line(self, image1, image2):
            yellow = detect_yellow(self,image1,image2)

            yellow_x = yellow[0]
            yellow_z = yellow[3]

            yellow_contours = detect_yellow_contours(image2)
            blue_contours = detect_blue_contours(image2)
            y2 = detect_blue(self, image1, image2)
            y2 = y2[3]



            y1, _ = get_y1_y2(yellow_contours, blue_contours)

            x1, y1 = yellow_x, int(y1)
            x2, y2 = yellow_x, int(y2)




            line_thickness = 2
            cv2.line(image2, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)


            cv2.circle(image2, (int(yellow_x), int(yellow_z)),1, (0, 0, 255), 2)

            cv2.imshow('zx', image2)



#Control Part starts here------------------------------------------------------------------------------------------

        #Estimate control inputs for open-loop control


        cv2.waitKey(1)



        fk_end_effector = get_EE_by_FK(self,0,0,0,np.pi/2 )
        vision_end_effector = end_effector_position(self, self.image1, self.image2)

        print(vision_end_effector)


        #draw_line(self, self.image1, self.image2)

#        self.vision_EF = Float64MultiArray()
#        self.vision_EF.data = vision_end_effector
        self.fk_EF=Float64MultiArray()
        self.fk_EF.data=fk_end_effector

        #print(vision_end_effector)

        # # Publishing the desired trajectory on a topic named trajectory(for lab 3)

        try:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.image1, "bgr8"))
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.image2, "bgr8"))

#            self.vision_end_effector_pub.publish((self.vision_EF))
            self.fk_end_effector_pub.publish((self.fk_EF))




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
