import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
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

        # Grid yapılandırması
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Video Frame
        self.video_frame = tk.Frame(root, bg="#34495e", bd=5)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.video_label = tk.Label(self.video_frame, bg="#34495e")
        self.video_label.pack()

        # Buttons Frame
        self.buttons_frame = tk.Frame(root, bg="#2c3e50")
        self.buttons_frame.grid(row=1, column=0, pady=10)

        self.stage1_button = tk.Button(self.buttons_frame, text="Stage 1", command=lambda: self.stage_action(1),
                                       bg="#1abc9c", fg="white", font=("Helvetica", 12), width=10)
        self.stage1_button.grid(row=0, column=0, padx=10, pady=5)

        self.stage2_button = tk.Button(self.buttons_frame, text="Stage 2", command=lambda: self.stage_action(2),
                                       bg="#3498db", fg="white", font=("Helvetica", 12), width=10)
        self.stage2_button.grid(row=0, column=1, padx=10, pady=5)

        self.stage3_button = tk.Button(self.buttons_frame, text="Stage 3", command=lambda: self.stage_action(3),
                                       bg="#9b59b6", fg="white", font=("Helvetica", 12), width=10)
        self.stage3_button.grid(row=0, column=2, padx=10, pady=5)

        self.emergency_button = tk.Button(self.buttons_frame, text="Emergency", command=self.emergency_action,
                                          bg="red", fg="white", font=("Helvetica", 12), width=10)
        self.emergency_button.grid(row=0, column=3, padx=10, pady=5)

        # Video & detection systems
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.current_stage = None
        self.motion_detector = MotionDetection()
        self.object_detector = ObjectDetection()
        self.tracker = ObjectTracker()
        self.color_detector = ColorDetection()
        self.shape_detector = ShapeDetection()
        self.shape_detection_enabled = False

        self.update_video()

    def stage_action(self, stage):
        self.current_stage = stage
        if stage == 1:
            self.object_detector.detection_enabled = True
            self.shape_detection_enabled = False
        elif stage == 2:
            self.object_detector.detection_enabled = True
            self.shape_detection_enabled = False
        elif stage == 3:
            self.object_detector.detection_enabled = True
            self.shape_detection_enabled = True
            print("Stage 3: Object + shape detection enabled")

    def update_video(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("❌ Kamera görüntüsü alınamadı.")
            self.root.after(10, self.update_video)
            return

        if self.current_stage == 1:
            # 🚨 Motion Detection Stage
            motion_boxes = self.motion_detector.detect(frame)
            for (x, y, w, h) in motion_boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "Motion", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        elif self.current_stage == 2:
            # 🎯 Object Detection + Friend/Foe
            frame, detections = self.object_detector.process_frame(frame)
            self.tracker.update(detections)
            self.tracker.draw(frame)

            for (x1, y1, x2, y2, conf, cls) in detections:
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                color_name = self.detect_color_at_center(frame, center_x, center_y)
                status = "enemy" if color_name == "red" else "friend" if color_name == "blue" else ""
                if status:
                    cv2.putText(frame, status, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif self.current_stage == 3 and self.shape_detection_enabled:
            # 🔷 Object + Shape Detection
            frame, detections = self.object_detector.process_frame(frame)
            for x1, y1, x2, y2, conf, cls in detections:
                bbox = (x1, y1, x2 - x1, y2 - y1)
                shape_detections, frame = self.shape_detector.detect(frame, bbox)

        # 🔁 Tkinter için çevir ve göster
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_video)


    def detect_color_at_center(self, frame, x, y):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_frame[y, x]
        for color_name, ranges in self.color_detector.color_ranges.items():
            for (lower, upper) in ranges:
                if all(np.array(lower) <= [h, s, v]) and all([h, s, v] <= np.array(upper)):
                    return color_name
        return "Unknown"

    def emergency_action(self):
        self.running = False
        self.cap.release()
        self.video_label.config(image="")
        messagebox.showwarning("Emergency", "Emergency action triggered!")

    def on_closing(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
