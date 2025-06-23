import cv2
from pose_module import PoseDetector
from eye_contact_module import EyeContactDetector

def main():
    cap = cv2.VideoCapture(1)
    pose_detector = PoseDetector()
    eye_detector = EyeContactDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pose_frame = pose_detector.detect(frame.copy())
        eye_frame = eye_detector.detect(frame.copy())

        cv2.imshow('Pose Detection', pose_frame)
        cv2.imshow('Eye Contact Detection', eye_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pose_detector.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
