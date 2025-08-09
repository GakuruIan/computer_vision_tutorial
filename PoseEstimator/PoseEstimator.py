import mediapipe as mp
import cv2
import math
import numpy as np

class PoseEstimator:
    def __init__(self,mode=False,smoothLandmarks=True,detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smoothLandmarks = smoothLandmarks
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        # Right arm
        self.counter_right = 0
        self.stage_right = None

        # Left arm
        self.counter_left = 0
        self.stage_left = None
        

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
        self.landmarks_list = []
        
        if results.pose_landmarks:
            for id,lm in enumerate(results.pose_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if annotate:
                    cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                    self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                self.landmarks_list.append((id, cx, cy))
        return frame, self.landmarks_list
    
    def find_angle(self,frame,p1,p2,p3,annotate=True):
        _,x1,y1 = self.landmarks_list[p1]
        _,x2,y2 = self.landmarks_list[p2]
        _,x3,y3 = self.landmarks_list[p3]
        
        # getting angle
        angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2,x1-x2)) 
        
        if angle < 0:
            angle += 360
        
        if annotate:
            cv2.line(frame,(x1,y1),(x2,y2),(255,255,255,3))
            cv2.line(frame,(x2,y2),(x3,y3),(255,255,255,3))
            cv2.circle(frame,(x1,y1),7,(0, 255, 0),cv2.FILLED)
            cv2.circle(frame,(x1,y1),10,(0, 255, 0),1)
            cv2.circle(frame,(x2,y2),7,(0, 255, 0),cv2.FILLED)
            cv2.circle(frame,(x2,y2),10,(0, 255, 0),1)
            cv2.circle(frame,(x3,y3),7,(0, 255, 0),cv2.FILLED)
            cv2.circle(frame,(x3,y3),10,(0, 255, 0),1)
        return angle
    
    def curl_counter(self,frame):
        
        angle_r = self.find_angle(frame,12,14,16)
        
        angle_l =self.find_angle(frame,11,13,15)
        
        per_r = np.interp(angle_r,(170,330),(0,100))
        
        per_l = np.interp(angle_l,(170,335),(0,100))
        
        
        print({"Angle ":angle_r,"percentage ":per_r})
        
        if per_r == 100:
            self.stage_right = "down"
        if per_r < 40 and self.stage_right == "down":
            self.stage_right = "up"
            self.counter_right += 1
        
        if per_l == 100:
            self.stage_left = "down"
            
        if per_l < 40 and self.stage_left == "down":
            self.stage_left = "up"
            self.counter_left += 1
        
        cv2.putText(frame, f'Right Arm: {self.counter_right}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
        cv2.putText(frame, f'Left Arm: {self.counter_left}', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)