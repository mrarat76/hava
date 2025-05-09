import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import threading
import numpy as np
from motion_detection import MotionDetection
from model import ObjectDetection
from track import ObjectTracker
from color_detection import ColorDetection
from shape_detection import ShapeDetection

class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KoruCam")
        self.root.geometry("800x600")
        self.root.configure(bg="#2c3e50")

        # Video Frame
# Video Frame
        self.video_frame = tk.Frame(root, bg="#34495e", bd=5)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.video_label = tk.Label(self.video_frame, bg="#34495e")
        self.video_label.pack()

# Buttons Frame
        self.buttons_frame = tk.Frame(root, bg="#2c3e50")
        self.buttons_frame.grid(row=1, column=0, pady=10)

# Grid yapılandırması
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)


        self.stage1_button = tk.Button(
            self.buttons_frame, text="Stage 1",
            command=lambda: self.stage_action(1),
            bg="#1abc9c", fg="white", font=("Helvetica", 12), width=10
        )
        self.stage1_button.grid(row=0, column=0, padx=10, pady=5)

        self.stage2_button = tk.Button(
            self.buttons_frame, text="Stage 2",
            command=lambda: self.stage_action(2),
            bg="#3498db", fg="white", font=("Helvetica", 12), width=10
        )
        self.stage2_button.grid(row=0, column=1, padx=10, pady=5)

        self.stage3_button = tk.Button(
            self.buttons_frame, text="Stage 3",
            command=lambda: self.stage_action(3),
            bg="#9b59b6", fg="white", font=("Helvetica", 12), width=10
        )
        self.stage3_button.grid(row=0, column=2, padx=10, pady=5)

        self.emergency_button = tk.Button(
            self.buttons_frame, text="Emergency",
            command=self.emergency_action,
            bg="red", fg="white", font=("Helvetica", 12), width=10
        )
        self.emergency_button.grid(row=0, column=3, padx=10, pady=5)

        # Video Capture
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.motion_detector = MotionDetection()
        self.object_detector = ObjectDetection()
        self.tracker = ObjectTracker()
        self.color_detector = ColorDetection()
        self.shape_detector = ShapeDetection()
        self.shape_detection_enabled = False

        # Start video loop
        self.update_video()

    def update_video(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("❌ Kamera karesi alınamadı.")
            self.root.after(10, self.update_video)
            return

        # Get frame dimensions
        height, width = frame.shape[:2]
        center_region_size = 300  # Size of the center region square

        if self.shape_detection_enabled:
            # Define center region
            x1 = width//2 - center_region_size//2
            y1 = height//2 - center_region_size//2
            x2 = x1 + center_region_size
            y2 = y1 + center_region_size
            
            # Draw center region rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            # Extract and process center region
            center_region = frame[y1:y2, x1:x2]
            processed_region, shape_detections = self.shape_detector.detect(center_region)
            if processed_region is None or processed_region.shape != center_region.shape:
                processed_region = np.zeros_like(center_region)
                shape_detections = []

            frame[y1:y2, x1:x2] = processed_region

            # Draw shape names and colors
            for shape, color, (cx, cy) in shape_detections:
                adjusted_cx = x1 + cx
                adjusted_cy = y1 + cy
                color_name = self.detect_color_at_center(frame, adjusted_cx, adjusted_cy)
                text = f"{shape.upper()} ({color})"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(frame,
                            (adjusted_cx - text_width//2 - 5, adjusted_cy - 10),
                            (adjusted_cx + text_width//2 + 5, adjusted_cy + text_height + 10),
                            (0, 0, 0), -1)
                cv2.putText(frame, text,
                            (adjusted_cx - text_width//2, adjusted_cy + text_height//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if self.object_detector.detection_enabled:
            frame, detections = self.object_detector.process_frame(frame)
            self.tracker.update(detections)
            self.tracker.draw(frame)

            for (x1, y1, x2, y2, conf, cls) in detections:
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                color_name = self.detect_color_at_center(frame, center_x, center_y)
                status = ""
                if color_name == "red":
                    status = "enemy"
                elif color_name == "blue":
                    status = "friend"

                if status:
                    cv2.putText(frame, status, (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except Exception as e:
            print(f"⚠️ Görüntü dönüştürme hatası: {e}")

        self.root.after(10, self.update_video)


    def stage_action(self, stage):
        if stage == 1:
            self.object_detector.toggle_detection()
            self.shape_detection_enabled = False
        elif stage == 2:
            self.object_detector.toggle_detection()
            self.shape_detection_enabled = False
            if self.object_detector.detection_enabled:
                ret, frame = self.cap.read()
                if ret:
                    frame, detections = self.object_detector.process_frame(frame)
                    for (x1, y1, x2, y2, conf, cls) in detections:
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        color_name = self.detect_color_at_center(frame, center_x, center_y)
                        
                        # Add status label based on color
                        status = ""
                        if color_name == "red":
                            status = "enemy"
                        elif color_name == "blue":
                            status = "friend"
                        
                        if status != 1:
                            # Position the status text above the bounding box
                            cv2.putText(frame, status, (x1, y1 - 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 5)
                        
                        print(f"Detected color at center: {color_name}, Status: {status}, Confidence: {conf}, Class: {cls}")
        elif stage == 3:
            self.shape_detection_enabled = not self.shape_detection_enabled
            self.object_detector.detection_enabled = False
            print(f"Shape detection {'enabled' if self.shape_detection_enabled else 'disabled'}")
        else:      
            messagebox.showinfo("Stage Action", f"Stage {stage} action triggered!")

    def detect_color_at_center(self, frame, x, y):
        # Convert frame to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Get the HSV color at the center point
        pixel_color = hsv_frame[y, x]
        h, s, v = pixel_color

        # Check each color range
        for color_name, ranges in self.color_detector.color_ranges.items():
            for (lower, upper) in ranges:
                # Convert bounds to numpy arrays
                lower = np.array(lower)
                upper = np.array(upper)
                # Check if the pixel color falls within the current range
                if (lower[0] <= h <= upper[0] and 
                    lower[1] <= s <= upper[1] and 
                    lower[2] <= v <= upper[2]):
                    return color_name
        
        return "Unknown"

    def emergency_action(self):
        # Stop video and release resources
        self.running = False
        self.cap.release()
        self.video_label.config(image="")
        messagebox.showwarning("Emergency", "Emergency action triggered!")

    def on_closing(self):
        # Clean up on exit
        self.running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
