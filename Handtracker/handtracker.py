import mediapipe as mp 
import cv2

class HandTracker:
    def __init__(self,mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mp_hands = mp.solutions.hands
        #static mode:False -> Initial detection only once, then uses tracking to follow the hand across frames.
        #static mode:True -> Detects hands in every frame, useful for static images.
        #maxHands: Maximum number of hands to detect.
        #detectionCon: Minimum confidence for hand detection.
        #trackCon: Minimum confidence for hand tracking.
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode,
        max_num_hands=self.maxHands,
        min_detection_confidence=self.detectionCon,
        min_tracking_confidence=self.trackCon)
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame, annotate=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frameRGB)
        landmarks_list =[]

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id,lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if annotate:
                        cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                    landmarks_list.append((id, cx, cy,lm.z))
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
        return frame, landmarks_list