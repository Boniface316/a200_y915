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
        #Change this value to match yours
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

        #cv2.imshow("image1", image1)
        #cv2.imshow("image2", image2)

        target_location = get_target_location(self, centerX, centerY, centerZ1)
        target_location_meters = target_location*p

        return target_location_meters

    def actual_target_position(self):
        curr_time = np.array([rospy.get_time() - self.time_trajectory])
        x_d = float((2.5 * np.cos(curr_time * np.pi / 15))+0.5)
        y_d = float(2.5 * np.sin(curr_time * np.pi / 15))
        z_d = float((1 * np.sin(curr_time * np.pi / 15))+7.0)
        m = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        return np.array([x_d,y_d,z_d])
