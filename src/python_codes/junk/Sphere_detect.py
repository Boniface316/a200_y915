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
        #self.trajectory_pub = rospy.Publisher("trajectory", Float64MultiArray, queue_size=10)
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

        # Uncomment if you want to save the image
        # cv2.imwrite('image_copy.png', cv_image)

        def pixelTometer(self,image1, image2):
            circle1pos = detect_blue(self,image1, image2)
            z1 = 800 - circle1pos[3]
            circle2pos = detect_yellow(self,image1, image2)
            z2 = 800 - circle2pos[3]
            distance = z1 - z2
            return 2.5 / distance

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

        def initialize_detect_shape_var(template):
            startX = []
            startY = []
            endX = []
            endY = []
            w, h = template.shape[::-1]

            return startX, startY, endX, endY, w, h

        def get_resized_ratio(mask, scale):
            found = None
            resized = imutils.resize(mask, width = int(mask.shape[1] * scale))
            r = mask.shape[1]/float(resized.shape[1])
            return resized, r, found

        def get_maxVal_maxLoc(resized, template):
            res = cv2.matchTemplate(resized,template,cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
            return maxVal, maxLoc

        def append_X_Y(startX, startY, endX, endY,maxLoc, r, w, h):
            startX.append(int(maxLoc[0] * r))
            startY.append(int(maxLoc[1] * r))
            endX.append(int((maxLoc[0] + w) * r))
            endY.append(int((maxLoc[1] + h) * r))

            return startX, startY, endX, endY

        def detect_shape(self,mask, template):
            startX, startY, endX, endY, w, h = initialize_detect_shape_var(template)
            for scale in np.linspace(0.79, 1.0, 5)[::-1]:
                resized, r, found = get_resized_ratio(mask, scale)

                if resized.shape[0] < h or resized.shape[1] < w:
                    break

                maxVal, maxLoc = get_maxVal_maxLoc(resized, template)

                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, r)

                startX, startY, endX, endY = append_X_Y(startX, startY, endX, endY, maxLoc, r, w, h)

            centerX = (statistics.mean(endX) + statistics.mean(startX))/2
            centerY = (statistics.mean(endY) + statistics.mean(startY))/2

            return centerX, centerY

        def apply_mask_target(image):
            hsv_convert = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_convert, (10, 202, 0), (27, 255, 255))
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            return mask

        def get_target_location(self, centerX, centerY, centerZ):
            base_location = detect_yellow(self, self.image1, self.image2)
            base_location[2] = 800 - base_location[2]
            centerZ = 800 - centerZ
            target_location = (centerX - base_location[0], centerY - base_location[1], centerZ - base_location[2])
            target_location = np.asarray(target_location)
            return target_location

        def flying_object_location(self,image1,image2, template, threshold):
            mask1 = apply_mask_target(image1)
            mask2 = apply_mask_target(image2)

            centerY, centerZ1 = detect_shape(self, mask1, template)
            centerX, centerZ2 = detect_shape(self, mask2, template)

            p = pixelTometer(self, image1, image2)

            image1 = cv2.circle(image1, (int(centerY), int(centerZ1)), radius=2, color=(255, 255, 255), thickness=-1)
            image2 = cv2.circle(image2, (int(centerX), int(centerZ2)), radius=2, color=(255, 255, 255), thickness=-1)

            cv2.imshow("image1", image1)
            cv2.imshow("image2", image2)

            target_location = get_target_location(self, centerX, centerY, centerZ1)
            target_location_meters = target_location*p

            return target_location_meters


#------------------------------------------------------------------------------------------



        cv2.waitKey(1)
        # Publish the results
        # change te value of self.joint.data to your estimated value from thew images once you have finalized the code

        #
        # publish the estimated position of robot end-effector (for lab 3)


        # x_e_image=end_effector_position(self, self.image1,self.image2)
        # self.end_effector = Float64MultiArray()
        # self.end_effector.data = x_e_image

        # send control commands to joints (for lab 3)
        #q_d = control_closed(self, self.image1,self.image2)
        # q_d=control_open(self,self.image1,self.image2)
        # print("The joints after q_d",q_d)

        # print (calculate_forward_kinematics(self),EE_by_DH(self))
        self.joint1 = Float64()
        self.joint1.data = 0
        self.joint2 = Float64()
        self.joint2.data = 0
        self.joint3 = Float64()
        self.joint3.data = 0
        self.joint4 = Float64()
        self.joint4.data = 0

        # # Publishing the desired trajectory on a topic named trajectory(for lab 3)
        template = cv2.imread("/home/boniface/catkin_ws/src/ivr_assignment/src/images/cropped.png",0)


        x_d =flying_object_location(self,self.image1,self.image2, template, 0.8)   # getting the desired trajectory
        print(x_d)


        # self.trajectory_desired = Float64MultiArray()
        # self.trajectory_desired.data = x_d

        try:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.image1, "bgr8"))
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.image2, "bgr8"))
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
