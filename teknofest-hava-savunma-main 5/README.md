# KoruCam - Intelligent Surveillance System

KoruCam is an advanced computer vision-based surveillance system that combines object detection, tracking, color analysis, and shape detection capabilities. The system operates in three distinct stages, each offering specific functionality for surveillance and monitoring purposes.

## Dataset Link
Dataset Link: https://www.kaggle.com/datasets/serhiibiruk/balloon-object-detection

## Features

### Stage 1: Object Detection and Tracking
- Real-time object detection using YOLOv8 model
- Advanced object tracking using Kalman Filter
- Continuous tracking of detected objects with unique IDs
- Prediction of object trajectories

### Stage 2: Friend or Foe Identification
- Color-based object classification
- Identifies objects as:
  - Enemy (red objects)
  - Friend (blue objects)
- Real-time status display above detected objects

### Stage 3: Shape Analysis
- Detects and identifies basic geometric shapes in the center region:
  - Triangles
  - Squares
  - Circles
- Visual highlighting of detected shapes
- Shape classification display

## Technologies Used
- Python
- OpenCV (cv2)
- NumPy
- Tkinter (GUI)
- YOLOv12 (Object Detection)
- Kalman Filter (Object Tracking)

## Components
- `gui.py`: Main application interface and video processing
- `model.py`: YOLOv8-based object detection
- `track.py`: Kalman filter-based object tracking
- `color_detection.py`: Color analysis and classification
- `shape_detection.py`: Geometric shape detection
- `motion_detection.py`: Motion-based detection system

## Dependencies
- Python 3.x
- OpenCV
- NumPy
- Ultralytics YOLO
- PIL (Python Imaging Library)
- Tkinter

## Installation
1. Clone the repository
2. Install the required dependencies:
```bash
pip install opencv-python numpy ultralytics pillow
```

## Usage
1. Run the application:
```bash
python gui.py
```

2. Use the interface buttons to toggle different stages:
- Stage 1: Enable object detection and tracking
- Stage 2: Enable friend/foe identification
- Stage 3: Enable shape detection in the center region
- Emergency: Stop the application

## Emergency Stop
The system includes an emergency stop button that immediately halts all operations and closes the video feed.
