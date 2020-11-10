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

        #----------------------------------------------------------------------------------------------
        #Angle Detection starts here

        def angle_detection_blob(self,image1, image2):
            xyz_blob = utils2.create_xyz_table(self.image1, self.image2, "yellow")

            green_posn = xyz_blob.loc["green",]
            blue_posn = xyz_blob.loc["blue",]
            yellow_posn = xyz_blob.loc["yellow",]
            red_posn = xyz_blob.loc["red",]
            ja1_blob=0.1
            ja2_blob = np.arctan2((blue_posn[1] - green_posn[1]), -(blue_posn[2] - green_posn[2]))
            #ja2_blob = utils2.angle_normalization(ja2_blob)
            ja3_blob = np.arctan2((blue_posn[0] - green_posn[0]), (blue_posn[2] - green_posn[2]))
            #ja3_blob = utils2.angle_normalization(ja3_blob)
            ja4_blob = np.arctan2(green_posn[1] - red_posn[1], (green_posn[2] - red_posn[2]))-ja2_blob
            #ja4_blob = utils2.angle_normalization(ja4_blob) - ja2_blob
            return np.array([ja1_blob, ja2_blob, ja3_blob, ja4_blob])

        def angle_trajectory(self):
            curr_time = np.array([rospy.get_time() - self.time_trajectory])
            ja1 = 0.1
            ja2 = float((np.pi / 2) * np.sin((np.pi / 15) * curr_time))
            ja3 = float((np.pi / 2) * np.sin((np.pi / 18) * curr_time))
            ja4 = float((np.pi / 2) * np.sin((np.pi / 20) * curr_time))
            return np.array([ja1, ja2, ja3, ja4])

        def actual_target_position(self):
            curr_time = np.array([rospy.get_time() - self.time_trajectory])
            x_d = float((2.5 * np.cos(curr_time * np.pi / 15))+0.5)
            y_d = float(2.5 * np.sin(curr_time * np.pi / 15))
            z_d = float((1 * np.sin(curr_time * np.pi / 15))+7.0)
            m = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
            return np.array([x_d,y_d,z_d])

        #FK starts here--------------------------------------------------------------------------------

        def EE_by_DH(self):
            m=np.array([[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]])
            ja1, ja2, ja3, ja4 = angle_trajectory(self)
            T1 = np.array(
                [[np.cos(ja1), 0, np.sin(ja1), 0], [np.sin(ja1), 0, -np.cos(ja1), 0], [0, 1, 0, 2.5], [0, 0, 0, 1]])
            T2 = np.array(
                [[np.cos(ja2), 0, np.sin(ja2), 0], [np.sin(ja2), 0, -np.cos(ja2), 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            Tn=np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
            T3 = np.array(
                [[np.cos(ja3), 0, -np.sin(ja3), 3.5 * np.cos(ja3)], [np.sin(ja3), 0, np.cos(ja3), 3.5 * np.sin(ja3)],
                 [0, -1, 0, 0], [0, 0, 0, 1]])
            T4 = np.array(
                [[np.cos(ja4), -np.sin(ja4), 0, 3.0 * np.cos(ja4)], [np.sin(ja4), np.cos(ja4), 0, 3.0 * np.sin(ja4)],
                 [0, 0, 1, 0], [0, 0, 0, 1]])
            E1 = T1.dot(Tn)
            E1=E1.dot(T2)
            E2 = E1.dot(T3)
            EE = E2.dot(T4)
            #EE=EE.dot(m)
            # ee=EE[0:3,3]
            # ee=ee.reshape(1,3)
            return EE

        def calculate_forward_kinematics(self,image1,image2):
            ja1, ja2, ja3, ja4 = angle_trajectory(self)
            end_effector = np.array([3*np.cos(ja1+1.5708)*np.cos(ja2+1.5708)*np.cos(ja3)*np.cos(ja4)
              +3*np.sin(ja1+1.5708)*np.sin(ja3)*np.cos(ja4)
              -3*np.cos(ja1+1.5708)*np.sin(ja2+1.5708)*np.sin(ja4)
              +3.5*np.cos(ja1+1.5708)*np.cos(ja2+1.5708)*np.cos(ja3)
              +3.5*np.sin(ja1+1.5708)*np.sin(ja3),

             3*np.sin(ja1+1.5708)*np.cos(ja2+1.5708)*np.cos(ja3)*np.cos(ja4)
              -3*np.cos(ja1+1.5708)*np.sin(ja3)*np.cos(ja4)
              -3*np.sin(ja1+1.5708)*np.sin(ja2+1.5708)*np.sin(ja4)
            +3.5*np.sin(ja1+1.5708)*np.cos(ja2+1.5708)*np.cos(ja3)
            -3.5*np.cos(ja1+1.5708)*np.sin(ja3),

                 3 * np.sin(ja2+1.5708) * np.cos(ja3) * np.cos(ja4)
                 +3*np.cos(ja2+1.5708)*np.sin(ja4)+3.5*np.sin(ja2+1.5708)*np.cos(ja3)+2.5
             ])
            m=np.array([[0,1,0],[-1,0,0],[0,0,1]])

            return m.dot(end_effector)

        def end_effector_position(self, image1, image2):
            p = pixelTometer(self, image1, image2)
            m = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            cx, cy, cz, ct = p * (detect_yellow(self, image1, image2) - detect_red(self, image1, image2))
            return m.dot(np.array([cx, cy, cz]))

        def calculate_jacobian(self):
            ja1,ja2,ja3,ja4=angle_detection_blob(self,self.image1,self.image2)
            jacobian=np.array([[-3*np.sin(ja1)*np.cos(ja2)*np.cos(ja3)*np.cos(ja4)
                               -3.5*np.sin(ja1)*np.cos(ja2)*np.cos(ja3)
                               +3*np.cos(ja4)*np.sin(ja3)*np.cos(ja3)
                               +3.5*np.sin(ja3)*np.cos(ja1)
                               +3*np.sin(ja4)*np.sin(ja1)*np.sin(ja2),

                               -3*np.cos(ja1)*np.sin(ja2)*np.cos(ja3)*np.cos(ja4)
                               -3.5*np.cos(ja1)*np.sin(ja2)*np.cos(ja3)
                               -3*np.sin(ja4)*np.cos(ja1)*np.cos(ja2),

                               -3*np.cos(ja1)*np.cos(ja2)*np.sin(ja3)*np.cos(ja4)
                               -3.5*np.sin(ja3)*np.cos(ja1)*np.cos(ja2)
                               +3*np.cos(ja4)*np.cos(ja3)*np.sin(ja1)
                               +3.5*np.cos(ja3)*np.sin(ja1),

                               -3*np.cos(ja1)*np.cos(ja2)*np.cos(ja3)*np.sin(ja4)
                               -3*np.sin(ja4)*np.sin(ja3)*np.sin(ja1)
                              -3*np.cos(ja4)*np.cos(ja1)*np.sin(ja2)
                               ],
                              [

                                3*np.cos(ja3)*np.cos(ja4)*np.cos(ja1)*np.cos(ja2)
                               +3.5*np.cos(ja3)*np.cos(ja1)*np.cos(ja2)
                                +3*np.cos(ja4)*np.sin(ja3)*np.sin(ja1)
                                +3.5*np.sin(ja3)*np.sin(ja1),

                                -3*np.cos(ja3)*np.cos(ja4)*np.sin(ja1)*np.sin(ja2)
                                -3.5*np.cos(ja3)*np.sin(ja1)*np.sin(ja2)
                                -3*np.sin(ja4)*np.sin(ja1)*np.cos(ja2),

                                -3*np.sin(ja3)*np.cos(ja4)*np.sin(ja1)*np.cos(ja2)
                                -3.5*np.sin(ja3)*np.sin(ja1)*np.cos(ja2)
                                -3*np.cos(ja4)*np.cos(ja3)*np.cos(ja1)
                                -3.5*np.cos(ja3)*np.cos(ja1),

                                -3*np.cos(ja3)*np.sin(ja4)*np.sin(ja1)*np.cos(ja2)
                                +3*np.sin(ja4)*np.sin(ja3)*np.cos(ja1)
                                -3*np.cos(ja4)*np.sin(ja1)*np.sin(ja2)
                              ],
                              [ 0,

                                 3*np.cos(ja3)*np.cos(ja4)*np.cos(ja2)
                                 +3.5*np.cos(ja3)*np.cos(ja2)
                                 -3*np.sin(ja4)*np.sin(ja2),

                                 -3*np.sin(ja3)*np.cos(ja4)*np.sin(ja2)
                                 -3.5*np.sin(ja3)*np.sin(ja2),

                                  -3*np.cos(ja3)*np.sin(ja4)*np.sin(ja2)
                                  +3*np.cos(ja4)*np.cos(ja2)]

            ])
            return jacobian
        #Target Detection starts here---------------------------------------------------------------------

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
            mask = cv2.inRange(hsv_convert, (16, 69, 127), (19, 155, 213))
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


#Control Part starts here------------------------------------------------------------------------------------------

        #Estimate control inputs for open-loop control
        def control_open(self,image1,image2):
            # estimate time step
            cur_time = rospy.get_time()
            dt = cur_time - self.time_previous_step2
            self.time_previous_step2 = cur_time
            q =  angle_detection_blob(self,image1,image2)# estimate initial value of joints'
            J_inv = np.linalg.pinv(calculate_jacobian(self))  # calculating the psudeo inverse of Jacobian
            # desired trajectory
            pos_d = actual_target_position(self)
            # estimate derivative of desired trajectory
            self.error_d = (pos_d - self.error) / dt
            self.error = pos_d
            q_d = q + (dt * np.dot(J_inv, self.error_d.transpose()))  # desired joint angles to follow the trajectory
            return q_d

        def control_closed(self, image1,image2):
            # P gain
            K_p = np.array([[10, 0,0], [0,10, 0],[0,0,10]])
            # D gain
            K_d = np.array([[0.1, 0,0], [0,0.1, 0],[0,0,0.1]])
            # estimate time step
            cur_time = np.array([rospy.get_time()])
            dt = cur_time - self.time_previous_step
            self.time_previous_step = cur_time
            # robot end-effector position
            pos = end_effector_position(self,image1,image2)
            # desired trajectory
            pos_d = actual_target_position(self)
            # estimate derivative of error
            self.error_d = ((pos_d - pos) - self.error) / dt
            # estimate error
            self.error = pos_d - pos
            q = angle_detection_blob(self,image1,image2)  # estimate initial value of joints'
            J_inv = np.linalg.pinv(calculate_jacobian(self))  # calculating the psudeo inverse of Jacobian
            dq_d = np.dot(J_inv, (np.dot(K_d, self.error_d.transpose()) + np.dot(K_p,
                                                                                 self.error.transpose())))  # control input (angular velocity of joints)
            q_d = q + (dt * dq_d)  # control input (angular position of joints)
            return q_d

        cv2.waitKey(1)
        #Publishing data starts here-----------------------------
        # change te value of self.joint.data to your estimated value from thew images once you have finalized the code

        #
        # publish the estimated position of robot end-effector (for lab 3)


        # send control commands to joints (for lab 3)
        #q_d = control_closed(self, self.image1,self.image2)
        q_d=control_open(self,self.image1,self.image2)
        # print("Control angles",q_d)
        # print("Trajectory angles",angle_trajectory(self))
        # print("Blob angles",angle_detection_blob(self,self.image1,self.image2))
        template = cv2.imread("cropped.png", 0)
        print ("End eff",end_effector_position(self,self.image1,self.image2))
        print("FK pos",calculate_forward_kinematics(self,self.image1,self.image2))
        print(EE_by_DH(self))
        print("\n")

        self.joints = Float64MultiArray()
        self.joints.data = angle_trajectory(self)

        ja1,ja2,ja3,ja4=angle_trajectory(self)
        self.joint1 = Float64()
        self.joint1.data = ja1
        self.joint2 = Float64()
        self.joint2.data = ja2
        self.joint3 = Float64()
        self.joint3.data =ja3
        self.joint4 = Float64()
        self.joint4.data = ja4


        fk_end_effector = calculate_forward_kinematics(self, self.image1, self.image2)
        vision_end_effector = end_effector_position(self, self.image1, self.image2)
        self.vision_EF = Float64MultiArray()
        self.vision_EF.data = vision_end_effector
        self.fk_EF=Float64MultiArray()
        self.fk_EF.data=fk_end_effector

        # # Publishing the desired trajectory on a topic named trajectory(for lab 3)
        x_d =flying_object_location(self,self.image1,self.image2,template,0.8)
        x_a=actual_target_position(self)# getting the desired trajectory
        self.actual_target = Float64MultiArray()
        self.actual_target.data = x_a
        self.vision_target = Float64MultiArray()
        self.vision_target.data = x_d

        try:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.image1, "bgr8"))
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.image2, "bgr8"))

            self.vision_end_effector_pub.publish((self.vision_EF))
            self.fk_end_effector_pub.publish((self.fk_EF))




            self.actual_target_trajectory_pub.publish((self.actual_target))
            self.vision_target_trajectory_pub.publish((self.vision_target))

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

