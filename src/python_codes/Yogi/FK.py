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

        def detect_yellow(self,image1, image2):
            mask1 = cv2.inRange(image1, (0, 100, 100), (0, 255, 255))
            kernel = np.ones((5, 5), np.uint8)
            mask1 = cv2.dilate(mask1, kernel, iterations=3)
            M1 = cv2.moments(mask1)

            mask2 = cv2.inRange(image2, (0, 100, 100), (0, 255, 255))
            mask2 = cv2.dilate(mask2, kernel, iterations=3)
            M2 = cv2.moments(mask2)
            cy = int(M1['m10'] / M1['m00'])
            cz = int(M1['m01'] / M1['m00'])
            cx = int(M2['m10'] / M2['m00'])
            ct = int(M2['m01'] / M2['m00'])
            return np.array([cx, cy, cz, ct])

        def detect_red(self,image1, image2):
            mask1 = cv2.inRange(image1, (0, 0, 100), (0, 0, 255))
            kernel = np.ones((5, 5), np.uint8)
            mask1 = cv2.dilate(mask1, kernel, iterations=3)
            M1 = cv2.moments(mask1)

            mask2 = cv2.inRange(image2, (0, 0, 100), (0, 0, 255))
            mask2 = cv2.dilate(mask2, kernel, iterations=3)
            M2 = cv2.moments(mask2)
            cy = int(M1['m10'] / M1['m00'])
            cz = int(M1['m01'] / M1['m00'])
            cx = int(M2['m10'] / M2['m00'])
            ct = int(M2['m01'] / M2['m00'])
            return np.array([cx, cy, cz, ct])

        def detect_blue(self,image1, image2):
            mask1 = cv2.inRange(image1, (100, 0, 0), (255, 0, 0))
            kernel = np.ones((5, 5), np.uint8)
            mask1 = cv2.dilate(mask1, kernel, iterations=3)
            M1 = cv2.moments(mask1)

            mask2 = cv2.inRange(image2, (100, 0, 0), (255, 0, 0))
            mask2 = cv2.dilate(mask2, kernel, iterations=3)
            M2 = cv2.moments(mask2)
            cy = int(M1['m10'] / M1['m00'])
            cz = int(M1['m01'] / M1['m00'])
            cx = int(M2['m10'] / M2['m00'])
            ct = int(M2['m01'] / M2['m00'])

            return np.array([cx, cy, cz, ct])

        def detect_green(self,image1, image2):
            mask1 = cv2.inRange(image1, (0, 100, 0), (0, 255, 0))
            kernel = np.ones((5, 5), np.uint8)
            mask1 = cv2.dilate(mask1, kernel, iterations=3)
            M1 = cv2.moments(mask1)

            mask2 = cv2.inRange(image2, (0, 100, 0), (0, 255, 0))
            mask2 = cv2.dilate(mask2, kernel, iterations=3)
            M2 = cv2.moments(mask2)
            cy = int(M1['m10'] / M1['m00'])
            cz = int(M1['m01'] / M1['m00'])
            cx = int(M2['m10'] / M2['m00'])
            ct = int(M2['m01'] / M2['m00'])

            return np.array([cx, cy, cz, ct])

        def pixelTometer(self,image1, image2):
            circle1pos = detect_blue(self,image1, image2)
            z1 = 800 - circle1pos[3]
            circle2pos = detect_yellow(self,image1, image2)
            z2 = 800 - circle2pos[3]
            distance = z1 - z2
            return 2.5 / distance



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
            p = pixelTometer(self, image1, image2)
            m = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            yellow_posn = detect_yellow(self, image1, image2)
            red_posn = detect_red(self, image1, image2)
            yellow_posn[3] = 800 - yellow_posn[3]
            red_posn[3] = 800 - red_posn[3]
            cx, cy, cz1, cz2 = p * (red_posn - yellow_posn)
            return np.array([cx, cy, cz1, cz2])



#Control Part starts here------------------------------------------------------------------------------------------

        #Estimate control inputs for open-loop control


        cv2.waitKey(1)



        fk_end_effector = get_EE_by_FK(self,0,0,0,np.pi/2 )
        vision_end_effector = end_effector_position(self, self.image1, self.image2)

#        self.vision_EF = Float64MultiArray()
#        self.vision_EF.data = vision_end_effector
        self.fk_EF=Float64MultiArray()
        self.fk_EF.data=fk_end_effector

        print(vision_end_effector)

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
