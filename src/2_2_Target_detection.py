#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
import math
import numpy as np
import statistics
import imutils
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
        #self.end_effector_pub=rospy.Publisher("end_effector_prediction",Float64MultiArray,queue_size=10)
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
        self.actual_target_trajectory_pub = rospy.Publisher("actual_target_trajectory", Float64MultiArray, queue_size=10)
        self.vision_target_trajectory_pub = rospy.Publisher("vision_target_trajectory", Float64MultiArray, queue_size=10)
        self.time_trajectory = rospy.get_time()


        # initialize errors
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
        self.time_previous_step2 = np.array([rospy.get_time()], dtype='float64')
        # initialize error and derivative of error for trajectory tracking
        self.error = np.array([0.0, 0.0,0.0], dtype='float64')
        self.error_d = np.array([0.0, 0.0,0.0], dtype='float64')
        self.p2m = np.array([0.0], dtype='float64')
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


        def detect_blue(self,image1, image2):
            #gaussian blur is applied to reduce the noise and smoothen the image
            image_gau_blur1 = cv2.GaussianBlur(image1, (1, 1), 0)
            #image converted to HSV
            hsv1 = cv2.cvtColor(image_gau_blur1, cv2.COLOR_BGR2HSV)
            #lower and higher value of HSV are defined
            lower_red1 = np.array([70, 0, 0])
            higher_red1 = np.array([255, 255, 255])
            #apply colour threshold
            red_range1 = cv2.inRange(hsv1, lower_red1, higher_red1)
            #apply bitwise conversion
            res_red1 = cv2.bitwise_and(image_gau_blur1, image_gau_blur1, mask=red_range1)
            #convert the iamge to grey scale
            red_s_gray1 = cv2.cvtColor(res_red1, cv2.COLOR_BGR2GRAY)
            #Apply canny to detect edges
            canny_edge1 = cv2.Canny(red_s_gray1, 30, 70)
            #Find contours
            contours1, hierarchy1 = cv2.findContours(canny_edge1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #find the radius and center coordinates for the image
            (x1, y1), radius1 = cv2.minEnclosingCircle(contours1[0])
            cy, cz1 = (int(x1), int(y1))
            radius1 = int(radius1)

            #same as above but for image 2
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

        def detect_yellow(self,image1, image2):
            #same as detect_blue()
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
            #Same as detect_blue, except it returns the contours
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
            #Same as detect_yellow_contours()
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
            #find the top of the yellow blob
            y1 = np.min(yellow_contours, axis = 0)
            y1 = y1[0][1]
            y1 = y1[:,1]

            #find the bottom of the blue blob
            y2 = np.max(blue_contours, axis = 0)
            y2 = y2[0][1]
            y2 = y2[:,1]

            return y1, y2

        def pixelTometer(self, image1,image2):
            #find the contours of blue and yellow blobs
            yellow_contours = detect_yellow_contours(image2)
            blue_contours = detect_blue_contours(image2)
            y2 = detect_blue(self, image1, image2)
            y2 = y2[3]

            #find the positions of top of the yellow and the bottom of the blue
            y1, y2 = get_y1_y2(yellow_contours, blue_contours)

            #calculate the ratio
            p2m = 2.5/(y1 - y2)

            return p2m


        def initialize_detect_shape_var(template):
            #initiate variables
            startX = []
            startY = []
            endX = []
            endY = []
            w, h = template.shape[::-1]

            return startX, startY, endX, endY, w, h

        def get_resized_ratio(mask, scale):

            found = None
            #resize the image
            resized = imutils.resize(mask, width = int(mask.shape[1] * scale))
            #the ratio of resize
            r = mask.shape[1]/float(resized.shape[1])
            return resized, r, found

        def get_maxVal_maxLoc(resized, template):
            #get the top left corner of the image
            res = cv2.matchTemplate(resized,template,cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
            return maxVal, maxLoc

        def append_X_Y(startX, startY, endX, endY,maxLoc, r, w, h):
            #append the coordinates of the image
            startX.append(int(maxLoc[0] * r))
            startY.append(int(maxLoc[1] * r))
            endX.append(int((maxLoc[0] + w) * r))
            endY.append(int((maxLoc[1] + h) * r))

            return startX, startY, endX, endY

        def detect_shape(self,mask, template):
            #initialize variables
            startX, startY, endX, endY, w, h = initialize_detect_shape_var(template)
            #Vary the image size until it matches the template
            for scale in np.linspace(0.79, 1.0, 5)[::-1]:
                resized, r, found = get_resized_ratio(mask, scale)

                #break if the resized image is smaller than the template
                if resized.shape[0] < h or resized.shape[1] < w:
                    break

                maxVal, maxLoc = get_maxVal_maxLoc(resized, template)

                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, r)

                #get the start and end location of the image
                startX, startY, endX, endY = append_X_Y(startX, startY, endX, endY, maxLoc, r, w, h)

            #take the average to find the center
            centerX = (statistics.mean(endX) + statistics.mean(startX))/2
            centerY = (statistics.mean(endY) + statistics.mean(startY))/2

            return centerX, centerY

        def apply_mask_target(image):
            #convert the image to HSV
            hsv_convert = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            #apply threshold and create a mask
            mask = cv2.inRange(hsv_convert, (10, 69, 0), (27, 255, 255))
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            return mask

        def get_target_location(self, centerX, centerY, centerZ):
            #get the position of yellow
            base_location = detect_yellow(self, self.image1, self.image2)
            #change from pixel frame to camera frame
            base_location[2] = 800 - base_location[2]
            centerZ = 800 - centerZ
            #find the location of the target reference to the base in pixel
            target_location = (centerX - base_location[0], centerY - base_location[1], centerZ - base_location[2])
            target_location = np.asarray(target_location)
            target_location[2] = target_location[2]
            return target_location



        def flying_object_location(self,image1,image2, template, threshold):
            #apply mask
            mask1 = apply_mask_target(image1)
            mask2 = apply_mask_target(image2)

            #get the coordinates of the target
            centerY, centerZ1 = detect_shape(self, mask1, template)
            centerX, centerZ2 = detect_shape(self, mask2, template)

            #get the pixel to meter ratio

            try:
                p=pixelTometer(self,image1,image2)
                self.p2m = p
            except Exception as e:
                p = self.p2m

            image1 = cv2.circle(image1, (int(centerY), int(centerZ1)), radius=2, color=(255, 255, 255), thickness=-1)
            image2 = cv2.circle(image2, (int(centerX), int(centerZ2)), radius=2, color=(255, 255, 255), thickness=-1)

            #get the pixel location of the target
            target_location = get_target_location(self, centerX, centerY, centerZ1)
            #convert the location into meters
            target_location_meters = target_location*p
            #offset the z value of the target
            target_location_meters[2] = target_location_meters[2] + 1

            return target_location_meters

        def actual_target_position(self):
            #auto gerated values for the target
            curr_time = np.array([rospy.get_time() - self.time_trajectory])
            x_d = float((2.5 * np.cos(curr_time * np.pi / 15))+0.5)
            y_d = float(2.5 * np.sin(curr_time * np.pi / 15))
            z_d = float((1 * np.sin(curr_time * np.pi / 15))+7.0)
            return np.array([x_d,y_d,z_d])


#------------------------------------------------------------------------------------------

        cv2.waitKey(1)

        self.joint1 = Float64()
        self.joint1.data = 0
        self.joint2 = Float64()
        self.joint2.data = 0
        self.joint3 = Float64()
        self.joint3.data = 0
        self.joint4 = Float64()
        self.joint4.data = 0

        #load the template
        template = cv2.imread("cropped.png",0)

        #get the actual position of the target
        x_a=actual_target_position(self)
        #get the position of the targer using computer vision
        x_d =flying_object_location(self,self.image1,self.image2, template, 0.8)   # getting the desired trajectory
        #print(x_d)


        self.actual_target= Float64MultiArray()
        self.actual_target.data = x_a
        self.vision_target=Float64MultiArray()
        self.vision_target.data=x_d

        try:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.image1, "bgr8"))
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.image2, "bgr8"))
            self.robot_joint1_pub.publish(self.joint1)
            self.robot_joint2_pub.publish(self.joint2)
            self.robot_joint3_pub.publish(self.joint3)
            self.robot_joint4_pub.publish(self.joint4)
            self.actual_target_trajectory_pub.publish((self.actual_target))
            self.vision_target_trajectory_pub.publish((self.vision_target))
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
