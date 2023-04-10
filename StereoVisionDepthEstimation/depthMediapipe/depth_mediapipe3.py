import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Set up cameras
left_cam = cv2.VideoCapture(0)
right_cam = cv2.VideoCapture(1)

right_cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
left_cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# Set up mediapipe pose estimation
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        # Capture frames from both cameras
        ret_left, frame_left = left_cam.read()
        ret_right, frame_right = right_cam.read()

        if not ret_left or not ret_right:
            break

        # Resize frames
        frame_left = cv2.resize(frame_left, (640, 480))
        frame_right = cv2.resize(frame_right, (640, 480))

        # Calculate disparity map
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(gray_left, gray_right)
        disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Perform pose estimation on left frame
        results_left = pose.process(frame_left)

        if results_left.pose_landmarks:
            # Extract coordinates of left shoulder and right hip
            x_left_shoulder, y_left_shoulder, _ = [int(results_left.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame_left.shape[1]), int(results_left.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame_left.shape[0]), 0]
            x_right_hip, y_right_hip, _ = [int(results_left.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame_left.shape[1]), int(results_left.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame_left.shape[0]), 0]

            # Draw bounding box around person's body
            box_width = int(1.5 * abs(x_left_shoulder - x_right_hip))
            box_height = 2 * box_width
            x_left = int((x_left_shoulder + x_right_hip) / 2 - box_width / 2)
            y_top = int(y_left_shoulder - box_height / 2)
            cv2.rectangle(frame_left, (x_left, y_top), (x_left + box_width, y_top + box_height), (0, 255, 0), 2)

            # Calculate distance to person using bounding box dimensions and disparity map
            focal_length = 100
            baseline = 9
            box_center_x = x_left + box_width // 2
            box_center_y = y_top + box_height // 2
            disparity_at_center = disparity[box_center_y, box_center_x]
            if disparity_at_center > 0:
                depth = (focal_length * baseline) / disparity_at_center
                cv2.putText(frame_left, f"{depth:.2f} meters", (x_left, y_top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display frames
        cv2.imshow('Left Frame', frame_left)
        cv2.imshow('Left Frame', frame_left)
        cv2.imshow('Disparity Map', disparity)
        
        if cv2.waitKey(1) == ord('q'):
            break

#Release cameras and close windows

left_cam.release()
right_cam.release()
cv2.destroyAllWindows()
