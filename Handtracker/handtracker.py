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
        handedness_list =[]
        

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks,hand_handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                single_hand_landmarks = []
                for id,lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if annotate:
                       
                        cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                    single_hand_landmarks.append((id, cx, cy,lm.z))
                
                landmarks_list.append(single_hand_landmarks)
                handedness_label = hand_handedness.classification[0].label
                handedness_list.append(handedness_label)
                
                if annotate :
                    wrist_x,wrist_y = single_hand_landmarks[0][1],single_hand_landmarks[0][2]
                            
                    cv2.putText(frame,f"{hand_handedness.classification[0].label} hand",(wrist_x - 50, wrist_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2) 
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                
            
        return frame, landmarks_list,handedness_list