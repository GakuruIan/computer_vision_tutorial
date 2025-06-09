import mediapipe as mp
import cv2

class PoseEstimator:
    def __init__(self,mode=False,smoothLandmarks=True,detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smoothLandmarks = smoothLandmarks
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smoothLandmarks,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def process_frame(self, frame, annotate=True):
        frameRGB= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(frameRGB)
        landmarks_list = []
        
        if results.pose_landmarks:
            for id,lm in enumerate(results.pose_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if annotate:
                    cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                landmarks_list.append((id, cx, cy))
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return frame, landmarks_list