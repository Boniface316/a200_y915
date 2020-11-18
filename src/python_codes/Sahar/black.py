#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
import math
import utils2
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

        def contour_method1(self,image1, image2):
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image1_blur=cv2.medianBlur(gray1,3)

            cimg1 = cv2.cvtColor(image1_blur, cv2.COLOR_GRAY2BGR)
            circles1 = cv2.HoughCircles(image1_blur, cv2.HOUGH_GRADIENT, 1, image1.shape[0]/64,param1=45, param2=22, minRadius=0, maxRadius=0)
            print(("Circle1",circles1))
            if circles1 is not None:
                circles1 = np.uint16(np.around(circles1))
                for i in circles1[0, :]:
                    cv2.circle(cimg1, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(cimg1, (i[0], i[1]), 2, (0, 0, 255), 2)

            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            image2_blur = cv2.medianBlur(gray2, 3)

            cimg2 = cv2.cvtColor(image2_blur, cv2.COLOR_GRAY2BGR)
            circles2 = cv2.HoughCircles(image2_blur, cv2.HOUGH_GRADIENT, 1, 20, param1=45, param2=21, minRadius=0,
                                       maxRadius=0)
            print(("Circle 2",circles2))
            if circles2 is not None:
                circles2 = np.uint16(np.around(circles2))
                for i in circles2[0, :]:
                    cv2.circle(cimg2, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(cimg2, (i[0], i[1]), 2, (0, 0, 255), 2)
            cv2.imshow('detected circles1', cimg1)
            cv2.imshow('detected circles2', cimg2)


            return 0

        def contour_method2(self,image1,image2):
            image1_gray=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
            mask1=cv2.inRange(image1,(0,0,0),(180,255,30))
            kernel_erode=np.ones((3,3),np.uint8)
            kernel_dilate = np.ones((1, 1), np.uint8)
            mask1=cv2.erode(mask1,kernel_erode,iterations=5)
            mask1=cv2.dilate(mask1,kernel_dilate,iterations=2)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            opening = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)

            contours = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)[0]
            contours.sort(key=lambda x: cv2.boundingRect(x)[0])
            cv2.drawContours(image1, contours, -1, (255, 255, 255), 1)

            array = []
            ii = 1
            print len(contours)
            for c in contours:
                (x, y), r = cv2.minEnclosingCircle(c)
                center = (int(x), int(y))
                r = int(r)
                if r >= 6 and r <= 10:
                    cv2.circle(image1, center, r, (0, 255, 0), 2)
                    array.append(center)

            cv2.imshow("preprocessed", image1)
            cv2.imshow("mask",mask1)
            return 0

        def contour_method3(self,image1,image2):
            mask = cv2.inRange(image1, (0, 0, 0), (180, 255, 30))
            cv2.imshow("mask",mask)
            #mask = np.zeros(image2.shape, dtype=np.uint8)
            gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY )[1]
            kernel_erode = np.ones((3, 3), np.uint8)
            kernel_dilate = np.ones((1, 1), np.uint8)
            mask = cv2.erode(mask, kernel_erode, iterations=6)
            mask = cv2.dilate(mask, kernel_dilate, iterations=0)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            canny = cv2.Canny(mask, 100, 170)
            cv2.imshow("canny", canny)
            cimg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 20, param1=20, param2=20, minRadius=0,
                                       maxRadius=0)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 2)
            #cv2.imshow('cimg', cimg)
            return 0

        def contour_method4(self,image1,image2):
            mask = np.zeros(image2.shape, dtype=np.uint8)
            gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            #mask = cv2.inRange(image1, (0, 0, 0), (180, 255, 30))
            #gray=cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)[1]
            print(thresh)
            #Filter using contour hierarchy
            cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hierarchy = hierarchy[0]
            for component in zip(cnts, hierarchy):
                currentContour = component[0]
                currentHierarchy = component[1]
                x, y, w, h = cv2.boundingRect(currentContour)
                # Only select inner contours
                if currentHierarchy[3] > 0:
                    cv2.drawContours(mask, [currentContour], -1, (255, 255, 255), -1)

            # Filter contours on mask using contour approximation
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print (len(cnts))
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.05 * peri, True)
                if len(approx) > 5:
                    cv2.drawContours(mask, [c], -1, (0, 0, 0), -1)
                else:
                    cv2.drawContours(image1, [c], -1, (36, 255, 12), 2)

            cv2.imshow('thresh', thresh)
            cv2.imshow('image', image1)
            cv2.imshow('mask', mask)
            return 0

        def contour_method5(self,image1,image2):
            template=cv2.imread("cropped_blob.png")
            w = template.shape[0]
            h=template.shape[1]
            res= cv2.matchTemplate(image2,template,cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(image2, top_left, bottom_right, 255, 2)
            cv2.imshow("image1",image2)

            return 0

        # def pixelTometer(self,image1, image2):
        #     circle1pos = detect_blue(self,image1, image2)
        #     z1 = 800 - circle1pos[3]
        #     circle2pos = detect_yellow(self,image1, image2)
        #     z2 = 800 - circle2pos[3]
        #     distance = z1 - z2
        #     return 2.5 / distance

        #----------------------------------------------------------------------------------------------
        #Angle Detection starts here
        # def detect_angles_blob(self,image1,image2):
        #     p=pixelTometer(self,image1,image2)
        #     yellow=p*detect_yellow(self,image1,image2)
        #     blue=p*detect_blue(self,image1,image2)
        #     green=p*detect_green(self,image1,image2)
        #     red=p*detect_red(self,image1,image2)
        #     ja1=0.1
        #     ja2=np.pi/2-np.arctan2((blue[2] - green[2]), (blue[1] - green[1]))
        #     ja3 = np.arctan2((blue[3] - green[3]), (blue[0] - green[0]))-np.pi/2
        #     ja4 = np.arctan2((green[2] - red[2]), -(green[1] - red[1]))-np.pi/2-ja2
        #
        #     return np.array([ja1,ja2,ja3,ja4])

        def angle_trajectory(self):
            curr_time = np.array([rospy.get_time() - self.time_trajectory])
            ja1 = 0.1
            ja2 = float((np.pi / 2) * np.sin((np.pi / 15) * curr_time))
            ja3 = float((np.pi / 2) * np.sin((np.pi / 18) * curr_time))
            ja4 = float((np.pi / 2) * np.sin((np.pi / 20) * curr_time))
            return np.array([ja1, ja2, ja3, ja4])

        contour_method2(self,self.image1,self.image2)
        cv2.waitKey(1)

        self.joints = Float64MultiArray()
        #self.joints.data = detect_angles_blob(self,self.image1,self.image2)
        ja1,ja2,ja3,ja4=angle_trajectory(self)
        self.joint1 = Float64()
        self.joint1.data = ja1
        self.joint2 = Float64()
        self.joint2.data = ja2
        self.joint3 = Float64()
        self.joint3.data = ja3
        self.joint4 = Float64()
        self.joint4.data = ja4

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

