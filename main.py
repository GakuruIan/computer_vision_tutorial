import cv2
from Handtracker import HandTracker
from PoseEstimator import PoseEstimator

def main():
    video_path = 'videos/workout_session.mp4'
    cam = cv2.VideoCapture(video_path)
    # hand_tracker = HandTracker()
    pose_estimator = PoseEstimator()


    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    target_size = (640,400)  # (width, height)

    # fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    # out = cv2.VideoWriter('output/output.mp4', fourcc, 20.0, (frame_width, frame_height))

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break
        
        # cv2.flip(frame, 1, frame)
        # frame, landmarks = hand_tracker.process_frame(frame, annotate=True)
        frame =cv2.resize(frame, target_size)
        frame, pose_landmarks = pose_estimator.process_frame(frame, annotate=True)

        # out.write(frame)
        cv2.imshow('Pose Estimator', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cam.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()