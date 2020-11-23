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
        self.counter = np.array([0.0], dtype='float64')
        self.base_location = (0,0)
        self.ee_location = (0,0)
        self.green_location = (0.0)
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

        def angle_trajectory(self):
            curr_time = np.array([rospy.get_time() - self.time_trajectory])
            ja1 = 0.1
            ja2 = float((np.pi / 2) * np.sin((np.pi / 15) * curr_time))
            ja3 = float((np.pi / 2) * np.sin((np.pi / 18) * curr_time))
            ja4 = float((np.pi / 2) * np.sin((np.pi / 20) * curr_time))
            return np.array([ja1, ja2, ja3, ja4])

        def template_match_base(self, image1, image2, template):
            mask1 = cv2.inRange(image1, (0, 0, 0), (180, 255, 30))
            thresh1, dst1 = cv2.threshold(mask1, 5, 255, cv2.THRESH_BINARY_INV )

            w, h = template.shape[::-1]

            res = cv2.matchTemplate(dst1,template,cv2.TM_CCOEFF_NORMED)


            #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            #top_left = max_loc

            #bottom_right = (top_left[0] + w, top_left[1] + h)

            threshold = 0.6
            loc = np.where( res >= threshold)

            try:
                pt = (int(statistics.mean(loc[1])),int(statistics.mean(loc[0])))
                self.base_location = pt
            except Exception as e:
                pt  = self.base_location


            rectangle = np.array([[pt[0], pt[1]], [pt[0]+w, pt[1]], [pt[0]+w, pt[1]+h], [pt[0], pt[1]+h]])

            image1 = cv2.fillConvexPoly(image1, rectangle, (255, 0, 0))

            #cv2.rectangle(image1, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

            #cv2.rectangle(image1,top_left, bottom_right, 255, 2)




            return image1

        def ee_template_match(self, image1, image2, base_template, ee_template):
            image1 = template_match_base(self,self.image1,self.image2, base_template)

            mask1 = cv2.inRange(image1, (0, 0, 0), (180, 255, 30))
            thresh1, dst1 = cv2.threshold(mask1, 5, 255, cv2.THRESH_BINARY_INV )

            w, h = ee_template.shape[::-1]

            res = cv2.matchTemplate(dst1,ee_template,cv2.TM_CCOEFF_NORMED)


            #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            #top_left = max_loc

            #bottom_right = (top_left[0] + w, top_left[1] + h)

            threshold = 0.6
            loc = np.where( res >= threshold)

            try:
                pt = (int(statistics.mean(loc[1])),int(statistics.mean(loc[0])))
                self.ee_location = pt
            except Exception as e:
                pt  = self.ee_location


            rectangle = np.array([[pt[0], pt[1]], [pt[0]+2*w, pt[1]], [pt[0]+2*w, pt[1]+2*h], [pt[0], pt[1]+2*h]])

            image1 = cv2.fillConvexPoly(image1, rectangle, (255, 0, 0))

            #cv2.rectangle(image1, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

            #cv2.rectangle(image1,top_left, bottom_right, 255, 2)

            #cv2.imshow("ee", image1)

            return image1

        def create_boxes(pt,w,h):
            box_3 = (pt[0]-w, pt[1])
            box_0 = (box_3[0], box_3[1] - h)
            box_1 = (box_0[0]+w, box_0[1])
            box_2 = (box_1[0]+w, box_1[1])
            box_6 = (box_3[0], box_3[1] + h)
            box_5 = (pt[0]+w, pt[1])
            box_7 = (box_6[0]+w, box_6[1])
            box_8 = (box_7[0]+w, box_7[1])

            boxes = (box_0, box_1, box_2, box_3, pt, box_5, box_6, box_7, box_8)

            return boxes

        def draw_box(self, image1, image2, template, counter, base_template, ee_template):
            #incomplete code
            w, h = template.shape[::-1]

            if counter < 5:
                mask1 = cv2.inRange(image1, (0, 0, 0), (180, 255, 30))
                thresh1, dst1 = cv2.threshold(mask1, 5, 255, cv2.THRESH_BINARY_INV )

                res = cv2.matchTemplate(dst1,template,cv2.TM_CCOEFF_NORMED)

                threshold = 0.9
                loc = np.where( res >= threshold)

                try:
                    pt = (int(statistics.mean(loc[1])),int(statistics.mean(loc[0])))
                    self.green_location = pt
                except Exception as e:
                    pt  = self.green_location


            else:
                #image1 = template_match_base(self, image1, image2, base_template)
                image1 = ee_template_match(self, image1, image2, base_template, ee_template)

                boxes  = create_boxes(self.green_location, w, h)
                green_location = count_black_pixels(boxes, image1, template)
                self.green_location = green_location


            return 0

        def count_black_pixels(boxes, image, template):
            counts = []
            mask = cv2.inRange(image, (0, 0, 0), (180, 255, 30))
            thresh, dst = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY_INV )
            w, h = template.shape[::-1]

            for box in boxes:
                counts = []

                for box in boxes:
                    x_start = box[0]
                    y_start = box[1]
                    x_end = x_start + w
                    y_end = y_start + h
                    cropped_image = dst[y_start:y_end, x_start:x_end]

                    white_pixels_count = cv2.countNonZero(cropped_image)
                    total_number_of_pixel = cropped_image.shape[0]*cropped_image.shape[1]
                    black_pixel_count = total_number_of_pixel - white_pixels_count
                    counts.append(black_pixel_count)

                green_location_box = np.argmax(counts)
                box_location = boxes[green_location_box]
                x_start = box_location[0]
                y_start = box_location[1]
                x_end = x_start + w
                y_end = y_start + h
                center_of_box = (int((x_start+x_end)/2),int((y_start + y_end)/2))
                cv2.circle(image, center_of_box, 2, (255,0,255), 2)
                cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0,0,0), 2)
                cv2.imshow("box", image)
                return center_of_box






        self.joints = Float64MultiArray()
        #self.joints.data = detect_angles_blob(self,self.image1,self.image2)

        counter = self.counter


        if counter > 30:
            ja1,ja2,ja3,ja4=angle_trajectory(self)
            if counter > 200:
                counter = 0
        else:
            ja1 = 0
            ja2 = 0
            ja3 = 0
            ja4 = 0


        self.counter = counter + 1

        base_template = cv2.imread("/home/boniface/catkin_ws/src/ivr_assignment/src/python_codes/Yogi/full_base.png",0)
        ee_template = cv2.imread("/home/boniface/catkin_ws/src/ivr_assignment/src/python_codes/Yogi/ee_template_2.png", 0)
        middle_template = cv2.imread("/home/boniface/catkin_ws/src/ivr_assignment/src/python_codes/Yogi/middle_template.png", 0)


        cv2.waitKey(1)

        draw_box(self, self.image1, self.image2, middle_template, counter, base_template, ee_template)





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
