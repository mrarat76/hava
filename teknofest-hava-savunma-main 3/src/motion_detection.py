import cv2
import numpy as np

class MotionDetection:
    def __init__(self):
        self.motion_enabled = True
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()

    def toggle_motion_detection(self):
        self.motion_enabled = not self.motion_enabled

    def process_frame(self, frame):
        if not self.motion_enabled:
            return frame

        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)

        # Find contours of the moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles around detected motion
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small movements
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame