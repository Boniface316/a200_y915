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
        # initialize errors
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
        self.time_previous_step2 = np.array([rospy.get_time()], dtype='float64')
        # initialize error and derivative of error for trajectory tracking
        self.error = np.array([0.0, 0.0,0.0], dtype='float64')
        self.error_d = np.array([0.0, 0.0,0.0], dtype='float64')
        self.ee_location = (0, 0)
        self.counter = np.array([0.0], dtype='float64')
        self.ee_location_zx = (0,0)
        self.ee_location_zy = (0,0)
        self.base_zy = (0.0)
        self.base_zx = (0,0)
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

        #Template match for base starts here-------------------------------------------------------
        def find_blue_zy(self, image1, template):

            mask1 = cv2.inRange(image1, (0, 0, 0), (180, 255, 30))
            thresh1, dst1 = cv2.threshold(mask1, 5, 255, cv2.THRESH_BINARY_INV )

            w, h = template.shape

            res = cv2.matchTemplate(dst1,template,cv2.TM_CCOEFF_NORMED)

            threshold = 0.75
            loc = np.where( res >= threshold)

            pt = (int(statistics.mean(loc[1])),int(statistics.mean(loc[0])))
            rectangle = np.array([[pt[0], pt[1]], [pt[0]+w, pt[1]], [pt[0]+w, pt[1]+h], [pt[0], pt[1]+h]])


            radius = int(h/2)

            center_of_blue = (pt[0] + radius, pt[1] + radius)

            return center_of_blue

        def find_blue_zx(self, image2, template):

            mask1 = cv2.inRange(image2, (0, 0, 0), (180, 255, 30))
            thresh1, dst1 = cv2.threshold(mask1, 5, 255, cv2.THRESH_BINARY_INV )

            w, h = template.shape

            res = cv2.matchTemplate(dst1,template,cv2.TM_CCOEFF_NORMED)

            threshold = 0.75
            loc = np.where( res >= threshold)

            pt = (int(statistics.mean(loc[1])),int(statistics.mean(loc[0])))
            rectangle = np.array([[pt[0], pt[1]], [pt[0]+w, pt[1]], [pt[0]+w, pt[1]+h], [pt[0], pt[1]+h]])

            radius = int(h/2)

            center_of_blue = (pt[0] + radius, pt[1] + radius)

            return center_of_blue


    #Template Match for green starts here---------------------------------------------------------
        def template_match_for_green_zy(self,image1, template):
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            mask1 = cv2.inRange(image1, (0, 0, 0), (180, 255, 30))
            thresh1, dst1 = cv2.threshold(mask1, 5, 255, cv2.THRESH_BINARY_INV)

            for degrees in range(0, 360, 1):
                rotate = imutils.rotate_bound(template, degrees)
                w, h = rotate.shape[::-1]
                res = cv2.matchTemplate(dst1, rotate, cv2.TM_CCOEFF_NORMED)
                threshold = 0.45 #For EE make thsi 0.65 and for green make it 0.45 (keep tweaking it)
                loc = np.where(res >= threshold)

                for pt in zip(*loc[::-1]):

                    try:
                        pt = (int(statistics.mean(loc[1])), int(statistics.mean(loc[0])))
                        print(pt)
                        self.base_location = pt
                    except Exception as e:
                        pt = self.base_location
                    rectangle = np.array([[pt[0], pt[1]], [pt[0] + w, pt[1]], [pt[0] + w, pt[1] + h], [pt[0], pt[1] + h]])
                    # image1 = cv2.fillConvexPoly(image1, rectangle, (255, 0, 0))
                    cv2.rectangle(image1, (pt[0], pt[1]), (pt[0]+w, pt[1]+h), (255, 0, 0), 2)
                    # cv2.imshow("image_green1",image1)

                    return np.array([pt[0],pt[1]])



        def template_match_for_green_zx(self,image2, template):
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            mask1 = cv2.inRange(image2, (0, 0, 0), (180, 255, 30))
            thresh1, dst1 = cv2.threshold(mask1, 5, 255, cv2.THRESH_BINARY_INV)

            for degrees in range(0, 360, 1):
                rotate = imutils.rotate_bound(template, degrees)
                w, h = rotate.shape[::-1]
                res = cv2.matchTemplate(dst1, rotate, cv2.TM_CCOEFF_NORMED)
                threshold = 0.45 #For EE make thsi 0.65 and for green make it 0.45 (keep tweaking it)
                loc = np.where(res >= threshold)

                for pt in zip(*loc[::-1]):

                    try:
                        pt = (int(statistics.mean(loc[1])), int(statistics.mean(loc[0])))
                        print(pt)
                        self.base_location = pt
                    except Exception as e:
                        pt = self.base_location
                    rectangle = np.array([[pt[0], pt[1]], [pt[0] + w, pt[1]], [pt[0] + w, pt[1] + h], [pt[0], pt[1] + h]])
                    # image1 = cv2.fillConvexPoly(image1, rectangle, (255, 0, 0))
                    cv2.rectangle(image2, (pt[0], pt[1]), (pt[0]+w, pt[1]+h), (0, 0, 255), 2)
                    # cv2.imshow("image_green2",image2)

                    return np.array([pt[0],pt[1]])

        #Template Match for EE starts here------------------------------------------

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


        def is_not_empty(any_structure):
            try:
                rtn = any_structure[0]
                return True
            except Exception as e:
                return False

        def template_match_for_ee_zy(self,image1, template):
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            mask1 = cv2.inRange(image1, (0, 0, 0), (180, 255, 30))
            thresh1, dst1 = cv2.threshold(mask1, 5, 255, cv2.THRESH_BINARY_INV)

            for degrees in range(0, 10, 1):
                rotate = imutils.rotate_bound(template, degrees)
                w, h = rotate.shape[::-1]
                res = cv2.matchTemplate(dst1, rotate, cv2.TM_CCOEFF_NORMED)
                threshold = 0.95 #For EE make thsi 0.65 and for green make it 0.45 (keep tweaking it)
                loc = np.where(res >= threshold)



                loc_0_not_empty  = is_not_empty(loc[0])
                loc_1_not_empty  = is_not_empty(loc[1])

                if loc_0_not_empty == True and loc_1_not_empty == True:
                    pt = (int(statistics.mean(loc[1])), int(statistics.mean(loc[0])))
                    self.ee_location_zy = pt
                    break

                else:
                    try:
                        pt = detect_shape(self,dst1, template)
                        self.ee_location_zy = pt
                        break
                    except Exception as e:
                        raise


            x_start = int(self.ee_location_zy[0])
            y_start = int(self.ee_location_zy[1])

            pt = (x_start, y_start)
            # image1 = cv2.fillConvexPoly(image1, rectangle, (255, 0, 0))
            #rectangle = np.array([[pt[0], pt[1]], [pt[0] + w, pt[1]], [pt[0] + w, pt[1] + h], [pt[0], pt[1] + h]])
            #cv2.rectangle(image1, (pt[0], pt[1]), (pt[0]+w, pt[1]+h), (255, 0, 0), 2)
            image1 = cv2.circle(image1, pt, 8, (0,0,255), -1)
            #cv2.imshow("image_ee1",image1)

            return image1

        def template_match_for_ee_zx(self,image2, template):
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            mask1 = cv2.inRange(image2, (0, 0, 0), (180, 255, 30))
            thresh1, dst1 = cv2.threshold(mask1, 5, 255, cv2.THRESH_BINARY_INV)

            for degrees in range(0, 10, 1):
                rotate = imutils.rotate_bound(template, degrees)
                w, h = rotate.shape[::-1]
                res = cv2.matchTemplate(dst1, rotate, cv2.TM_CCOEFF_NORMED)
                threshold = 0.95 #For EE make thsi 0.65 and for green make it 0.45 (keep tweaking it)
                loc = np.where(res >= threshold)



                loc_0_not_empty  = is_not_empty(loc[0])
                loc_1_not_empty  = is_not_empty(loc[1])

                if loc_0_not_empty == True and loc_1_not_empty == True:
                    pt = (int(statistics.mean(loc[1])), int(statistics.mean(loc[0])))
                    self.ee_location_zy = pt
                    break

                else:
                    try:
                        pt = detect_shape(self,dst1, template)
                        self.ee_location_zy = pt
                        break
                    except Exception as e:
                        raise


            x_start = int(self.ee_location_zy[0])
            y_start = int(self.ee_location_zy[1])

            pt = (x_start, y_start)
                        # image1 = cv2.fillConvexPoly(image1, rectangle, (255, 0, 0))
                        #rectangle = np.array([[pt[0], pt[1]], [pt[0] + w, pt[1]], [pt[0] + w, pt[1] + h], [pt[0], pt[1] + h]])
                        #cv2.rectangle(image1, (pt[0], pt[1]), (pt[0]+w, pt[1]+h), (255, 0, 0), 2)
            image2 = cv2.circle(image2, pt, 8, (0,0,255), -1)
                        #cv2.imshow("image_ee1",image1)

            return image2


        #Angle Trajectory starts here----------------------------------------------

        def angle_trajectory(self):
            curr_time = np.array([rospy.get_time() - self.time_trajectory])
            ja1 = 0.1
            ja2 = float((np.pi / 2) * np.sin((np.pi / 15) * curr_time))
            ja3 = float((np.pi / 2) * np.sin((np.pi / 18) * curr_time))
            ja4 = float((np.pi / 2) * np.sin((np.pi / 20) * curr_time))
            return np.array([ja1, ja2, ja3, ja4])


        #Angle Detection starts here---------------------------------------------------------

        counter = self.counter

        template_for_base=cv2.imread("full_base.png",0)
        template_for_green=cv2.imread("green_template.png")
        template_for_EE=cv2.imread("ee_template.png")

        if counter > 30:
            ja1,ja2,ja3,ja4=angle_trajectory(self)
            if counter > 200:
                counter = 0

        else:
            ja1 = 0
            ja2 = 0
            ja3 = 0
            ja4 = 0
            if counter > 5:
                self.base_zy = find_blue_zy(self, self.image1, template_for_base)
                self.base_zx = find_blue_zy(self, self.image2, template_for_base)




        self.counter = counter + 1

        #
        #image2 = template_match_for_ee_zx(self, self.image2, template_for_EE)

        #cv2.imshow("image_ee2",image2)
        #cv2.imshow("image_ee1",image1)

        image1 = self.image1

        image1 = cv2.circle(image1, self.base_zy, 14, (255,0,0), -1)
        image1 = template_match_for_ee_zy(self, image1, template_for_EE)

        #cv2.imshow("blue-red-1", image1)


        image2 = self.image2

        image2 = cv2.circle(image2, self.base_zx, 14, (255,0,0), -1)
        image2 = template_match_for_ee_zx(self, image2, template_for_EE)

        #cv2.imshow("blue-red-2", image2)




        cv2.waitKey(1)

        self.joints = Float64MultiArray()
        #self.joints.data = detect_angles_blob(self,self.image1,self.image2)
        #ja1,ja2,ja3,ja4=angle_trajectory(self)
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