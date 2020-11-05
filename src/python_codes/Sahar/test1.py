import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import os
import utils2


import math
from angles import normalize_angle
class image_converter:
    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
        # initialize a publisher to send images from camera2 to a topic named image_topic2
        self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
        self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
        # initialize the bridge between openCV and ROS
        self.time_trajectory=rospy.get_time()
        self.bridge = CvBridge()
    # Recieve data, process it, and publish
    def callback2(self,data):
        # Recieve the image
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback1(self,data):
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        template = cv2.imread('cropped.png', 0)
        try:
            target_xy, target_angle = utils2.target_loc_angle(self.cv_image1, self.cv_image2, template, 0.2)
        except Exception as e:
            pass

        xyz_blob = utils2.create_xyz_table(self.cv_image1, self.cv_image2, "yellow")

        green_posn = xyz_blob.loc["green",]
        blue_posn = xyz_blob.loc["blue",]
        yellow_posn = xyz_blob.loc["yellow",]
        red_posn = xyz_blob.loc["red",]
        curr_time=np.array([rospy.get_time()-self.time_trajectory])
        ja1 = 0.1
        ja2=float((np.pi/2)*np.sin((np.pi/15)*curr_time))
        ja3 =float( (np.pi / 2) * np.sin((np.pi / 18) * curr_time))
        ja4 = float((np.pi / 2) * np.sin((np.pi / 20) * curr_time))

        T1=np.matrix([[np.cos(ja1),0,np.sin(ja1),0],[np.sin(ja1),0,-np.cos(ja1),0],[0,1,0,2.5],[0,0,0,1]])
        T2 = np.matrix([[np.cos(ja2), 0, np.sin(ja2), 0], [np.sin(ja2), 0, -np.cos(ja2), 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        T3 = np.matrix([[np.cos(ja3), 0, -np.sin(ja3), 3.5*np.cos(ja3)], [np.sin(ja3), 0, np.cos(ja3), 3.5*np.sin(ja3)], [0, -1, 0, 0], [0, 0, 0, 1]])
        T4 = np.matrix([[np.cos(ja4), -np.sin(ja4),0, 3.0 * np.cos(ja4)], [np.sin(ja4), np.cos(ja4),0, 3.0 * np.sin(ja4)],
              [0, 0, 1, 0], [0, 0, 0, 1]])

        end_effector = np.array([3 * np.cos(ja1) * np.cos(ja2) * np.cos(ja3) * np.cos(ja4)
                                 + 3 * np.sin(ja1) * np.sin(ja3) * np.cos(ja4)
                                 - 3 * np.cos(ja1) * np.sin(ja2) * np.sin(ja4)
                                 + 3.5 * np.cos(ja1) * np.cos(ja2) * np.cos(ja3)
                                 + 3.5 * np.sin(ja1) * np.sin(ja3),

                                 3 * np.sin(ja1) * np.cos(ja2) * np.cos(ja3) * np.cos(ja4)
                                 - 3 * np.cos(ja1) * np.sin(ja3) * np.cos(ja4)
                                 - 3 * np.sin(ja1) * np.sin(ja2) * np.sin(ja4)
                                 + 3.5 * np.sin(ja1) * np.cos(ja2) * np.cos(ja3)
                                 - 3.5 * np.cos(ja1) * np.sin(ja3),

                                 3 * np.sin(ja2) * np.cos(ja3) * np.cos(ja4)
                                 + 3 * np.cos(ja2) * np.sin(ja4) + 3.5 * np.sin(ja2) * np.cos(ja3) + 2.5
                                 ])
        E1=T1.dot(T2)
        E2=E1.dot(T3)
        EE=E2.dot(T4)
        end_eff=yellow_posn-red_posn
        #print(end_effector,EE)
        print(target_xy)
        



        ja2_blob = np.arctan2((blue_posn[1] - green_posn[1]), -(blue_posn[2] - green_posn[2]))
        #ja2_blob = utils2.angle_normalization(ja2_blob)
        ja3_blob = np.arctan2((blue_posn[0] - green_posn[0]), (blue_posn[2] - green_posn[2]))
        ja3_blob = utils2.angle_normalization(ja3_blob)
        ja4_blob = np.arctan2(green_posn[1] - red_posn[1], -(green_posn[2] - red_posn[2]))
        ja4_blob = utils2.angle_normalization(ja4_blob) - ja2_blob
        # print("blobangles", ja2,ja2_blob,ja3,ja3_blob, ja4,ja4_blob)

        self.joint1 = Float64()
        self.joint1.data = ja1
        self.joint2 = Float64()
        self.joint2.data = ja2
        self.joint3 = Float64()
        self.joint3.data = ja3
        self.joint4 = Float64()
        self.joint4.data = ja4
        try:
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