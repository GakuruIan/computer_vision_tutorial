import cv2
from Handtracker import HandTracker
from PoseEstimator import PoseEstimator
from FaceDetection import FaceDetector
from FaceMesh import Facemesh

from GestureVolumeControl.GestureHandler import GestureHandler
from FingerCounting.FingerCounter import FingerCounter
from PoseEstimator.PoseEstimator import PoseEstimator

from utils.resize_video import resize_with_aspect_ratio

def main():
    video_path = 'videos/curls.mp4'
    cam = cv2.VideoCapture(video_path)
    # hand_tracker = HandTracker()
    pose_estimator = PoseEstimator()
    # face_detector = FaceDetector()
    # face_mesh = Facemesh(max_num_faces=2)
    # gesture = GestureHandler()
    
    # fingerCounter =FingerCounter()
    

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    target_size = (640,480)  # (width, height)

    # fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    # out = cv2.VideoWriter('output/output.mp4', fourcc, 20.0, (frame_width, frame_height))

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break
        
        #cv2.flip(frame, 1, frame)
        frame = resize_with_aspect_ratio(frame, width=600, height=500)
        # frame, landmarks = hand_tracker.process_frame(frame, annotate=True)
        # frame = fingerCounter.CountFingers(frame)
        
        frame,lm_list = pose_estimator.process_frame(frame,annotate=False)
        
        if len(lm_list) != 0 :
         pose_estimator.curl_counter(frame)
        
        

        
        # out.write(frame)
        cv2.imshow('Pose Estimator', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cam.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()