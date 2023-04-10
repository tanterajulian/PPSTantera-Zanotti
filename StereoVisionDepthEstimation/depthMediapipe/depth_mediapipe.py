import cv2
import mediapipe as mp
import numpy as np

# Define constants for stereo vision
FOCAL_LENGTH = 7.8 # mm
BASELINE = 90.0 # mm

# Initialize cameras
left_camera = cv2.VideoCapture(0)
right_camera = cv2.VideoCapture(1)

right_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
left_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# Initialize Mediapipe Pose and Drawing utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_estimator:
    while True:
        # Read frames from both cameras
        ret_left, frame_left = left_camera.read()
        ret_right, frame_right = right_camera.read()
        
        #if not ret_left or not ret_right:
        #    break
        
        # Detect pose landmarks using Mediapipe
        results_left = pose_estimator.process(frame_left)
        results_right = pose_estimator.process(frame_right)
        
        # Draw bounding boxes around body parts
        mp_drawing.draw_landmarks(frame_left, results_left.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame_right, results_right.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results_right.pose_landmarks is None:
                print("No landmarks detected in the right image")
                continue

        # Calculate depth for each body part
        for i, landmark in enumerate(results_left.pose_landmarks.landmark):
            # Get pixel coordinates of body part in left and right images
            x_left, y_left, _ = int(landmark.x * frame_left.shape[1]), int(landmark.y * frame_left.shape[0]), 0
            x_right, y_right, _ = int(results_right.pose_landmarks.landmark[i].x * frame_right.shape[1]), int(results_right.pose_landmarks.landmark[i].y * frame_right.shape[0]), 0

            if results_right.pose_landmarks is None:
                print("No landmarks detected in the right image")
                continue

            # Calculate disparity
            disparity = abs(x_left - x_right)
            
            # Calculate depth using stereo vision formula, skip if disparity is zero
            if disparity == 0:
                continue
            depth = (FOCAL_LENGTH * BASELINE) / disparity
            
            # Draw bounding box and depth label on left image
            cv2.rectangle(frame_left, (x_left-10, y_left-10), (x_left+10, y_left+10), (0, 255, 0), 2)
            cv2.putText(frame_left, '{:.1f}cm'.format(depth), (x_left-20, y_left-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display frames
        cv2.imshow('Left camera', frame_left)
        cv2.imshow('Right camera', frame_right)
        
        # Exit on key press
        if cv2.waitKey(1) == ord('q'):
            break
    
    left_camera.release()
    right_camera.release()
    cv2.destroyAllWindows()
