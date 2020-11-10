#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
import math
import utils2
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
        self.end_effector_pub=rospy.Publisher("end_effector_prediction",Float64MultiArray,queue_size=10)
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
        self.trajectory_desired_pub = rospy.Publisher("trajectory_desired", Float64MultiArray, queue_size=10)
        self.actual_target_pos_pub = rospy.Publisher("actual_target_position", Float64MultiArray, queue_size=10)
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
            circle2pos = detect_green(self,image1, image2)
            distance = np.sum((circle1pos - circle2pos) ** 2)
            return 3.5 / np.sqrt(distance)

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

        def EE_by_DH(self):
            m=np.array([[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]])
            ja1, ja2, ja3, ja4 = np.array([0,0,0,0])
            T1 = np.array(
                [[np.cos(ja1), 0, np.sin(ja1), 0], [np.sin(ja1), 0, -np.cos(ja1), 0], [0, 1, 0, 2.5], [0, 0, 0, 1]])
            T2 = np.array(
                [[np.cos(ja2), 0, np.sin(ja2), 0], [np.sin(ja2), 0, -np.cos(ja2), 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            T3 = np.array(
                [[np.cos(ja3), 0, -np.sin(ja3), 3.5 * np.cos(ja3)], [np.sin(ja3), 0, np.cos(ja3), 3.5 * np.sin(ja3)],
                 [0, -1, 0, 0], [0, 0, 0, 1]])
            T4 = np.array(
                [[np.cos(ja4), -np.sin(ja4), 0, 3.0 * np.cos(ja4)], [np.sin(ja4), np.cos(ja4), 0, 3.0 * np.sin(ja4)],
                 [0, 0, 1, 0], [0, 0, 0, 1]])
            E1 = T1.dot(T2)
            E2 = E1.dot(T3)
            EE = E2.dot(T4)
            EE=EE.dot(m)
            # ee=EE[0:3,3]
            # ee=ee.reshape(1,3)
            return EE

        def calculate_forward_kinematics(self,image1,image2):
            ja1, ja2, ja3, ja4 = np.array([0,0,0,0])
            end_effector = np.array([3*np.cos(ja1)*np.cos(ja2)*np.cos(ja3)*np.cos(ja4)
              +3*np.sin(ja1)*np.sin(ja3)*np.cos(ja4)
              -3*np.cos(ja1)*np.sin(ja2)*np.sin(ja4)
              +3.5*np.cos(ja1)*np.cos(ja2)*np.cos(ja3)
              +3.5*np.sin(ja1)*np.sin(ja3),

             3*np.sin(ja1)*np.cos(ja2)*np.cos(ja3)*np.cos(ja4)
              -3*np.cos(ja1)*np.sin(ja3)*np.cos(ja4)
              -3*np.sin(ja1)*np.sin(ja2)*np.sin(ja4)
            +3.5*np.sin(ja1)*np.cos(ja2)*np.cos(ja3)
            -3.5*np.cos(ja1)*np.sin(ja3),

                 3 * np.sin(ja2) * np.cos(ja3) * np.cos(ja4)
                 +3*np.cos(ja2)*np.sin(ja4)+3.5*np.sin(ja2)*np.cos(ja3)+2.5
             ])
            m=np.array([[1,0,0],[0,-1,0],[0,0,1]])
            end_effector=end_effector
            return m.dot(end_effector)

        def end_effector_position(self, image1, image2):
            p = pixelTometer(self, image1, image2)
            # m = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
            cx, cy, cz, ct = p * (detect_yellow(self, image1, image2) - detect_red(self, image1, image2))
            # img = cv2.circle(image1, (cy, cz), 2, (255, 255, 255), -1)
            # cv2.imshow("image1",img)
            return np.array([cx, cy, ct])

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


        def detect_shape(self,contour):
            shape = "Unknown"
            peri = cv2.arcLength(contour, True)
            #print("perimeter of the shape", peri)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            if len(approx) == 4 and  peri<45:
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
                if ar<=1.05:
                    M = cv2.moments(approx)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    shape = "rectangle"
            else: shape="circle"
            M = cv2.moments(approx)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            #if ar >= 0.95 and ar <= 1.05 else "rectangle"
            return  np.array([shape,cx, cy])

        def flying_object_location(self,image1,image2):
            hsv_convert_1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
            hsv_convert_2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv_convert_1, (16, 69, 127), (19, 155, 213))
            mask2 = cv2.inRange(hsv_convert_2, (16, 69, 127), (19, 155, 213))
            kernel = np.ones((2, 2), np.uint8)
            mask1 = cv2.dilate(mask1, kernel, iterations=1)
            mask2 = cv2.dilate(mask2, kernel, iterations=1)
            contours1, _ = cv2.findContours(mask1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contours2, _ = cv2.findContours(mask2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image1, contours1, -1, (0, 0, 255), 2)
            cv2.drawContours(image2, contours2, -1, (0, 0, 255), 2)
            p = pixelTometer(self, image1, image2)

            for i in range(len(contours1)):
                shape,cy,cz1 = detect_shape(self, contours1[i])
                if(shape=="circle"):
                    shape_new1,cy,cz1=shape,cy,cz1
            for i in range(len(contours2)):
                shape, cx, cz2 = detect_shape(self, contours2[i])
                if (shape == "circle"):
                    shape_new2,cx,cz2=shape,cx,cz2

            cx_=float(cx)*p
            cy_=float(cy)*p
            cz1_=float(cz1)*p
            cz2_=float(cz2)*p
            loc=np.array([cx_,cy_,cz1_,cz2_])
            x,y,z,t=p*((detect_yellow(self,self.image1,self.image2)-loc))
            image = cv2.circle(image2, (int(cx), int(cz2)), radius=2, color=(255, 255, 255), thickness=-1)
            cv2.imshow("image", image)
            return np.array([x,y,z])


#------------------------------------------------------------------------------------------

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
        # Publish the results
        # change te value of self.joint.data to your estimated value from thew images once you have finalized the code

        #
        # publish the estimated position of robot end-effector (for lab 3)


        # send control commands to joints (for lab 3)
        #q_d = control_closed(self, self.image1,self.image2)
        q_d=control_open(self,self.image1,self.image2)


        # print("Control angles",q_d)
        # print("Trajectory angles",angle_trajectory(self))
        # print("Blob angles",angle_detection_blob(self,self.image1,self.image2))
        print ("End eff",end_effector_position(self,self.image1,self.image2))
        print("FK pos",calculate_forward_kinematics(self,self.image1,self.image2))
        print(EE_by_DH(self))
        print("\n")

        self.joints = Float64MultiArray()
        self.joints.data = angle_trajectory(self)

        ja1,ja2,ja3,ja4=angle_detection_blob(self,self.image1,self.image2)
        self.joint1 = Float64()
        self.joint1.data = ja1
        self.joint2 = Float64()
        self.joint2.data = ja2
        self.joint3 = Float64()
        self.joint3.data =ja3
        self.joint4 = Float64()
        self.joint4.data = ja4


        x_e = calculate_forward_kinematics(self, self.image1, self.image2)
        x_e_image = end_effector_position(self, self.image1, self.image2)
        self.end_effector = Float64MultiArray()
        self.end_effector.data = x_e_image #x_e or x_e_image

        # # Publishing the desired trajectory on a topic named trajectory(for lab 3)
        x_d =flying_object_location(self,self.image1,self.image2)   # getting the desired trajectory
        self.trajectory_desired = Float64MultiArray()
        self.trajectory_desired.data = x_d

        self.target_actual=Float64MultiArray()
        x_a=actual_target_position(self)
        self.target_actual.data=x_a

        try:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.image1, "bgr8"))
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.image2, "bgr8"))

            self.end_effector_pub.publish((self.end_effector))
            self.trajectory_desired_pub.publish(self.trajectory_desired)
            self.actual_target_pos_pub.publish(self.target_actual)

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

