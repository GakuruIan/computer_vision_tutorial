from Handtracker import HandTracker
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import cv2
import math

class GestureHandler:
    def __init__(self):
        self.hand_tracker = HandTracker(detectionCon=0.75)
        
        
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = interface.QueryInterface(IAudioEndpointVolume)
        
        self.volume_range = self.volume.GetVolumeRange()
        self.min_volume = self.volume_range[0]
        self.max_volume = self.volume_range[1]
        

    def get_gesture(self,frame):
        frame,landmarks_list = self.hand_tracker.process_frame(frame,annotate=False)
        
        if(len(landmarks_list) != 0):
            thumb_tip = landmarks_list[4]
            index_tip = landmarks_list[8]
            
            # Use index finger length as reference (more stable)
            index_mcp = landmarks_list[5]  # Index finger MCP joint
            index_length = math.sqrt((index_tip[1] - index_mcp[1])**2 + 
                                    (index_tip[2] - index_mcp[2])**2 + 
                                    ((index_tip[3] - index_mcp[3])*1000)**2)
            
            # Calculate pinch distance
            pinch_distance = math.sqrt((thumb_tip[1] - index_tip[1])**2 + 
                                    (thumb_tip[2] - index_tip[2])**2 + 
                                    ((thumb_tip[3] - index_tip[3])*1000)**2)
            
            # Normalize by finger length
            normalized_pinch = pinch_distance / index_length
                 
            
            x1,y1 = landmarks_list[4][1], landmarks_list[4][2]
            x2,y2 = landmarks_list[8][1], landmarks_list[8][2]
            
            cx,cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            cv2.circle(frame, (x1, y1), 10, (255, 0, 255), -1)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 255), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)
            
            vol = np.interp(normalized_pinch, [0.1, 1.5], [self.min_volume, self.max_volume])
            
            
            init_vol =np.interp(normalized_pinch, [0.08, 1.5], [400, 50])
            init_vol = np.clip(init_vol, 50, 400)

            vol = np.clip(vol, self.min_volume, self.max_volume)
            self.volume.SetMasterVolumeLevel(vol, None)
     
            if normalized_pinch < 0.35:
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                
        
        bar_x1,bar_y1 =50,80
        bar_x2,bar_y2 = 85,400
        bar_height = bar_y2 - bar_y1
        
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (0, 255, 0), 2)
        
        current_vol = self.volume.GetMasterVolumeLevel()
        vol_percent = (current_vol - self.min_volume) / (self.max_volume - self.min_volume)
        vol_percent = np.clip(vol_percent, 0, 1)
        
        filled_height = int(vol_percent * bar_height)
        fill_y = bar_y2 - filled_height
        cv2.rectangle(frame, (bar_x1, fill_y), (bar_x2, bar_y2), (0, 255, 0), cv2.FILLED)
        
        cv2.putText(frame, f'Volume: {int(vol_percent * 100)}%', (bar_x1, bar_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        