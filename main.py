import cv2
from Handtracker import HandTracker
from PoseEstimator import PoseEstimator
from FaceDetection import FaceDetector
from FaceMesh import Facemesh

from GestureVolumeControl.GestureHandler import GestureHandler

def main():
    video_path = 'videos/face2.mp4'
    cam = cv2.VideoCapture(0)
    # hand_tracker = HandTracker()
    # pose_estimator = PoseEstimator()
    # face_detector = FaceDetector()
    # face_mesh = Facemesh(max_num_faces=2)
    gesture = GestureHandler()

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    target_size = (640,400)  # (width, height)

    # fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    # out = cv2.VideoWriter('output/output.mp4', fourcc, 20.0, (frame_width, frame_height))

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break
        
        cv2.flip(frame, 1, frame)
        # frame, landmarks = hand_tracker.process_frame(frame, annotate=True)
        gesture.get_gesture(frame)


        # out.write(frame)
        cv2.imshow('Face Mesh', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cam.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()