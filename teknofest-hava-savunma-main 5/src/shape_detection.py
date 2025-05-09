import cv2
import numpy as np

class ShapeDetection:
    def __init__(self):
        # HSV aralıklarını tanımla (mavi için genişlettik)
        self.lower_ranges = {
            'red1':  np.array([  0, 100, 100]),
            'red2':  np.array([160, 100, 100]),
            'green': np.array([ 35, 100, 100]),
            'blue':  np.array([ 80,  50,  50])   # eskiden [100,100,100]
        }
        self.upper_ranges = {
            'red1':  np.array([ 10, 255, 255]),
            'red2':  np.array([180, 255, 255]),
            'green': np.array([ 85, 255, 255]),
            'blue':  np.array([140, 255, 255])   # eskiden [130,255,255]
        }
        # Morfolojik işlemler için küçük çekirdek
        self.kernel = np.ones((3,3), np.uint8)

    def detect(self, frame, bbox=None):
        """
        Detect shapes in frame, optionally within a specific bounding box
        :param frame: BGR image
        :param bbox: Optional (x, y, w, h) bounding box to detect shapes within
        :return: List of (shape, color, (cx, cy)) detections and annotated frame
        """
        working_frame = frame.copy()
        if bbox is not None:
            x, y, w, h = bbox
            # Extract region of interest
            roi = working_frame[y:y+h, x:x+w]
            if roi.size == 0:  # Check if ROI is valid
                return [], frame
        else:
            roi = working_frame
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        detections = []

        # Kırmızı için iki parçayı OR ile birleştir
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, self.lower_ranges['red1'], self.upper_ranges['red1']),
            cv2.inRange(hsv, self.lower_ranges['red2'], self.upper_ranges['red2'])
        )
        masks = {
            'red':   mask_red,
            'green': cv2.inRange(hsv, self.lower_ranges['green'], self.upper_ranges['green']),
            'blue':  cv2.inRange(hsv, self.lower_ranges['blue'],  self.upper_ranges['blue'])
        }

        for color_name, mask in masks.items():
            # Improve mask quality with morphological operations
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < 1000:  # Increased minimum area for better detection
                    continue

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)  # Adjusted epsilon for better shape approximation

                shape = None
                if len(approx) == 3:
                    shape = 'triangle'
                elif len(approx) == 4:
                    x_r, y_r, w_r, h_r = cv2.boundingRect(approx)
                    ar = w_r / float(h_r)
                    fill_ratio = area / float(w_r*h_r)
                    if 0.8 <= ar <= 1.2 and fill_ratio > 0.7:
                        shape = 'square'
                    else:
                        shape = 'triangle'
                elif len(approx) > 4:
                    (x_c, y_c), radius = cv2.minEnclosingCircle(cnt)
                    circle_area = np.pi * radius * radius
                    if abs(area - circle_area) < 0.25 * circle_area:
                        shape = 'circle'

                if shape:
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        
                        # Draw the shape with thicker lines and color based on shape type
                        if shape == 'triangle':
                            color_bgr = (0, 0, 255)  # Red
                        elif shape == 'square':
                            color_bgr = (255, 0, 0)  # Blue
                        else:  # circle
                            color_bgr = (0, 255, 0)  # Green
                            
                        cv2.drawContours(roi, [cnt], -1, color_bgr, 2)
                        
                        # Add shape label inside the ROI
                        label = f"{shape}-{color_name}"
                        cv2.putText(roi, label, (cx-20, cy-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
                        
                        # If we're working within a bbox, adjust coordinates for return value
                        if bbox is not None:
                            detections.append((shape, color_name, (cx + bbox[0], cy + bbox[1])))
                        else:
                            detections.append((shape, color_name, (cx, cy)))

        # If we cropped the frame, copy the annotated region back
        if bbox is not None:
            x, y, w, h = bbox
            frame[y:y+h, x:x+w] = roi

        return detections, frame


