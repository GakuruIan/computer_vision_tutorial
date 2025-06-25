from Handtracker import HandTracker
import cv2 

class FingerCounter:
    def __init__(self):
       self.handTracker = HandTracker(detectionCon=0.75)
       
    def CountFingers(self,frame):
        image,landmarklist,handedness_list = self.handTracker.process_frame(frame,annotate=True)
        
        all_fingers_count = []
        total_fingers_up=0
        
        # in the list below we exclude the thumb
        fingerTipsId =[8, 12, 16, 20]
 
        if len(landmarklist) != 0:
            for hand_landmarks, handlabel in zip(landmarklist, handedness_list):
                fingers_count = []

                # Thumb (horizontal logic based on hand)
                if handlabel == 'Right':
                    if hand_landmarks[4][1] < hand_landmarks[3][1]:
                        fingers_count.append(1)
                    else:
                        fingers_count.append(0)
                else:
                    if hand_landmarks[4][1] > hand_landmarks[3][1]:
                        fingers_count.append(1)
                    else:
                        fingers_count.append(0)

                # Other 4 fingers (vertical logic)
                for tip_id in fingerTipsId:
                    if hand_landmarks[tip_id][2] < hand_landmarks[tip_id - 2][2]:
                        fingers_count.append(1)
                    else:
                        fingers_count.append(0)

                total_fingers = sum(fingers_count)
                total_fingers_up += total_fingers
                all_fingers_count.append({
                    "hands": handlabel,
                    "fingers_up": total_fingers
                })
        
        
        
        y_offset = 40
        for hand in all_fingers_count:
            label = f"{hand['hands']} Hand: {hand['fingers_up']}"
            cv2.putText(image, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 40
            
        cv2.putText(image, f"Total Fingers: {total_fingers_up}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return image
            