import cv2
import numpy as np

class ColorDetection:
    def __init__(self):
        # Define HSV color ranges for common colors
        self.color_ranges = {
            'red': [((0, 120, 70), (10, 255, 255)),
                    ((170, 120, 70), (180, 255, 255))],
            'green': [((36, 25, 25), (86, 255, 255))],
            'blue': [((94, 80, 2), (126, 255, 255))]
        }

    def detect(self, frame, color_name):
        """
        Detect regions of a specified color and draw bounding boxes.
        :param frame: BGR image (numpy array)
        :param color_name: One of the keys in self.color_ranges
        :return: Annotated frame, list of bounding boxes [(x, y, w, h), ...]
        """
        if color_name not in self.color_ranges:
            raise ValueError(f"Color '{color_name}' not supported.")

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masks = []
        for (lower, upper) in self.color_ranges[color_name]:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            masks.append(cv2.inRange(hsv, lower, upper))

        # Combine masks for colors with multiple ranges
        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_or(mask, m)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        # Find contours and draw bounding boxes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                bboxes.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, color_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame, bboxes
