pt = (200,400)
w = 20
h = 10


box_4 = (pt[0]-w, pt[1])
box_1 = (box_4[0], box_4[1] - h)
box_2 = (box_1[0]+w, box_1[1])
box_3 = (box_2[0]+w, box_2[1])
box_7 = (box_4[0], box_4[1] + h)
box_6 = (pt[0]+w, pt[1])
box_8 = (box_7[0]+w, box_7[1])
box_9 = (box_8[0]+w, box_8[1])

boxes = (box_1, box_2, box_3, box_4, pt, box_6, box_7, box_8, box_9)

boxes[0]

for box in boxes:
    print(box)



import cv2

img = cv2.imread("/home/boniface/catkin_ws/src/ivr_assignment/src/python_codes/Yogi/base_template.png")

img.shape[0] * img.shape[1]
