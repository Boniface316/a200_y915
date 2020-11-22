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

        def contour_method1(self,image1, image2):
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image1_blur=cv2.medianBlur(gray1,3)

            cimg1 = cv2.cvtColor(image1_blur, cv2.COLOR_GRAY2BGR)
            circles1 = cv2.HoughCircles(image1_blur, cv2.HOUGH_GRADIENT, 1, image1.shape[0]/64,param1=40, param2=22, minRadius=0, maxRadius=0)
            # print(("Circle1",circles1))
            if circles1 is not None:
                circles1 = np.uint16(np.around(circles1))
                for i in circles1[0, :]:
                    cv2.circle(cimg1, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(cimg1, (i[0], i[1]), 2, (0, 0, 255), 2)
                    # if i[2]==13:
                    #     c_blue_z1=i[0]
                    #     c_blue_y=i[1]
                    #     print("bluezy",c_blue_z1,c_blue_y)
                    #
                    # elif i[2]==8:
                    #     c_red_z1=i[0]
                    #     c_red_y=i[1]
                    # print("red zy",c_red_z1,c_red_y)



            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            image2_blur = cv2.medianBlur(gray2, 3)

            cimg2 = cv2.cvtColor(image2_blur, cv2.COLOR_GRAY2BGR)
            circles2 = cv2.HoughCircles(image2_blur, cv2.HOUGH_GRADIENT, 1, 20, param1=45, param2=25, minRadius=0,
                                       maxRadius=0)
            # print(("Circle 2",circles2))
            if circles2 is not None:
                circles2 = np.uint16(np.around(circles2))
                for j in circles2[0, :]:
                    cv2.circle(cimg2, (j[0], j[1]), j[2], (0, 255, 0), 2)
                    cv2.circle(cimg2, (j[0], j[1]), 2, (0, 0, 255), 2)
                    if j[2]==13:
                        c_blue_z2=j[0]
                        c_blue_x=j[1]
                        print("blue zx",c_blue_z2,c_blue_x)

                    elif i[2]==8:
                        c_red_z2=j[0]
                        c_red_x=j[1]
                        print ("red zx",c_red_z2,c_red_x)

            # print(cx,cz1)
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
            mask1 = cv2.inRange(image2, (0, 0, 0), (180, 255, 30))
            cv2.imshow("mask",mask1)
            #mask = np.zeros(image2.shape, dtype=np.uint8)
            gray1 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            thresh1 = cv2.threshold(gray1, 5, 255, cv2.THRESH_BINARY )[1]
            kernel_erode1 = np.ones((3, 3), np.uint8)
            kernel_dilate1 = np.ones((1, 1), np.uint8)
            mask1 = cv2.erode(mask1, kernel_erode1, iterations=6)
            mask1= cv2.dilate(mask1, kernel_dilate1, iterations=0)

            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            opening1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel1)
            canny1 = cv2.Canny(mask1, 100, 170)
            cv2.imshow("canny1", canny1)
            cimg1 = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
            cnts1 ,hierarchy1 = cv2.findContours(canny1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            (x0, y0), radius0 = cv2.minEnclosingCircle(cnts1[0])
            cx1, cz1 = (int(x0), int(y0))
            radius0 = int(radius0)
            print(cx1,cz1)
            # (x1, y1), radius1 = cv2.minEnclosingCircle(cnts1[1])
            # cx2, cz2 = (int(x1), int(y1))
            # radius0 = int(radius1)

            print(len(cnts1))

            # circles = cv2.HoughCircles(canny1, cv2.HOUGH_GRADIENT, 1, 20, param1=20, param2=20, minRadius=0,
            #                            maxRadius=0)
            # if circles is not None:
            #     circles = np.uint16(np.around(circles))
            #     for i in circles[0, :]:
            #         cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            #         cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 2)
            # # #cv2.imshow('cimg', cimg)
            return 0

        def contour_method4(self,image1,image2):
            #mask_ = cv2.bitwise_not(mask)
            mask = np.zeros(image2.shape, dtype=np.uint8)
            gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

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

            template=cv2.imread("base_template.png")
            w = template.shape[0]
            h=template.shape[1]
            res= cv2.matchTemplate(image2,template,cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(image2, top_left, bottom_right, 255, 2)
            cv2.imshow("image1",image2)

            return 0




        def angle_trajectory(self):
            curr_time = np.array([rospy.get_time() - self.time_trajectory])
            ja1 = 0.1
            ja2 = float((np.pi / 2) * np.sin((np.pi / 15) * curr_time))
            ja3 = float((np.pi / 2) * np.sin((np.pi / 18) * curr_time))
            ja4 = float((np.pi / 2) * np.sin((np.pi / 20) * curr_time))
            return np.array([ja1, ja2, ja3, ja4])



        def contour_method6(self,image1,image2):
            hsv=cv2.cvtColor(image1,cv2.COLOR_BGR2HSV)
            mask=cv2.inRange(hsv,(0,0,0),(250,250,0))

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            canny = cv2.Canny(opening, 150, 150)
            # canny = cv2.Canny(mask, 30, 70)
            cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(self.image1, cnts, -1, (255, 255, 255), -1)
            #print(len(cnts))

            cv2.imshow("canny", canny)
            cv2.imshow("img", self.image1)
            return 0



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

            if counter < 20:
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


                boxes = create_boxes(pt,w,h)


                cv2.imshow("middle", dst)


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
            if counter > 500:
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

        #template_match_base(self, self.image1, self.image2, base_template)
        #image1 = template_match_base(self,self.image1,self.image2, base_template)

        #image1 = ee_template_match(self, self.image1, self.image2, base_template, ee_template)

        #contour_method4(self,image1,self.image2)
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
