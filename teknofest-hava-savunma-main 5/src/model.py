from ultralytics import YOLO
import cv2

class ObjectDetection:
    def __init__(self):
        self.model = YOLO('/Users/mehdiarat/Downloads/best.pt')  # Load YOLOv8n model
        self.detection_enabled = False

    def toggle_detection(self):
        self.detection_enabled = not self.detection_enabled
        print(f"Detection Enabled: {self.detection_enabled}")  # Debug: Check if detection is toggled

    def process_frame(self, frame):
        # Eğer tespit kapalıysa, boş listeyle birlikte frame'i geri dön
        if not self.detection_enabled:
            return frame, []

        detections = []
        results = self.model(frame)  # inference
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                conf = float(box.conf[0])
                
                if conf < 0.4:
                    continue
                cls = int(box.cls[0])
                label = f"{self.model.names[cls]} {conf:.2f}"

                # koordinat sınırlarını güvence altına al
                h, w, _ = frame.shape
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))

                # kutu ve etiket çizimi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                detections.append((x1, y1, x2, y2, conf, cls))

        return frame, detections