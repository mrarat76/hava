import cv2
import numpy as np

class KalmanFilter2D:
    def __init__(self, dt=1.0, 
                 process_noise=1e-2, 
                 measurement_noise=1e-1):
        # 1) Dynamic (A) matrix
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)
        # 2) Measurement (H) matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        # 3) Noise covariance (Q, R)
        #    Q: process noise, R: measurement noise
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        # 4) Initial error covariance P₀
        P0 = np.eye(4, dtype=np.float32)
        P0[2,2] = P0[3,3] = 1.0   # leave the velocity components a bit uncertain initially
        self.kf.errorCovPost = P0
        self.initialized = False

    def init(self, x, y):
        """Assign initial measurement to the state."""
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True

    def predict(self):
        """Predict the next state."""
        return self.kf.predict()

    def correct(self, x, y):
        """Incorporate new measurement into correction step."""
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        return self.kf.correct(meas)

    def predict_future(self, steps):
        """Use matrix A to compute future state as A^n·statePost."""
        state = self.kf.statePost.flatten()
        A = self.kf.transitionMatrix
        future = np.linalg.matrix_power(A, steps) @ state
        return future  # [x, y, vx, vy]

class ObjectTracker:
    def __init__(self, match_threshold=50, dt=1.0, max_age=5):
        self.tracks = {}         # track_id -> KalmanFilter2D instance
        self.track_age = {}      # track_id -> number of frames since last matched
        self.next_id = 0
        self.match_thr = match_threshold
        self.dt = dt
        self.max_age = max_age   # frames to wait before deletion

    def update(self, detections):
        # 1) Predict and increment ages
        preds = {}
        for tid, kf2d in self.tracks.items():
            preds[tid] = kf2d.predict()[:2].astype(int)
            self.track_age[tid] = self.track_age.get(tid, 0) + 1

        new_tracks = {}
        new_ages   = {}

        # 2) Match detections and reset age
        for x1, y1, x2, y2, _, _ in detections:
            cx, cy = (x1 + x2)//2, (y1 + y2)//2

            best_id, best_dist = None, float('inf')
            for tid, (px, py) in preds.items():
                d = np.hypot(px - cx, py - cy)
                if d < best_dist and d < self.match_thr:
                    best_dist, best_id = d, tid

            if best_id is None:
                # new track
                kf2d = KalmanFilter2D(dt=self.dt)
                kf2d.init(cx, cy)
                tid = self.next_id
                self.next_id += 1
            else:
                # correct existing track
                kf2d = self.tracks[best_id]
                kf2d.correct(cx, cy)
                tid = best_id

            new_tracks[tid] = kf2d
            new_ages[tid] = 0
            preds.pop(tid, None)

        # 3) Keep unmatched old tracks up to max_age
        for tid, kf2d in self.tracks.items():
            age = self.track_age.get(tid, 0)
            if tid not in new_tracks and age < self.max_age:
                # can continue predicting forward but retain age
                kf2d.predict()
                new_tracks[tid] = kf2d
                new_ages[tid]   = age

        self.tracks    = new_tracks
        self.track_age = new_ages

    def draw(self, frame, steps_ahead=10):
        for tid, kf2d in self.tracks.items():
            # 1) Backup current state
            state_backup = kf2d.kf.statePost.copy()
            cov_backup   = kf2d.kf.errorCovPost.copy()
            
            # 2) Predict steps_ahead into future
            pred = None
            for _ in range(steps_ahead):
                pred = kf2d.kf.predict()
            
            # 3) Restore original state
            kf2d.kf.statePost     = state_backup
            kf2d.kf.errorCovPost  = cov_backup

            # 4) Draw a point at the predicted future state
            x, y = int(pred[0]), int(pred[1])
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(frame, f"ID:{tid}",
                        (x + 10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
