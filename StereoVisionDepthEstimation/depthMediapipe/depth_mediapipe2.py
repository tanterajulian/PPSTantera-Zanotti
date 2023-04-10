import cv2
import mediapipe as mp
import numpy as np

# set up mediapipe pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# set up cameras
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

cap_right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap_left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# set camera properties
width = 640
height = 480
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# set up mediapipe pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# set up focal length and baseline
focal_length = 800
baseline = 90

# loop through frames
while True:
    # read frames from cameras
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    

    # flip frames horizontally
    frame_left = cv2.flip(frame_left, 1)
    frame_right = cv2.flip(frame_right, 1)

    # detect pose landmarks in left and right images
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_left, mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_right:
        # process left image
        results_left = pose_left.process(frame_left)
        if results_left.pose_landmarks is not None:
            # draw landmarks on left image
            mp_drawing.draw_landmarks(frame_left, results_left.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # get coordinates of person's nose
            x_left, y_left, _ = int(results_left.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * frame_left.shape[1]), int(results_left.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * frame_left.shape[0]), 0

            # draw box around person's nose
            cv2.rectangle(frame_left, (x_left - 30, y_left - 30), (x_left + 30, y_left + 30), (0, 255, 0), 2)

        # process right image
        results_right = pose_right.process(frame_right)
        if results_right.pose_landmarks is not None:
            # draw landmarks on right image
            mp_drawing.draw_landmarks(frame_right, results_right.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # get coordinates of person's nose
            x_right, y_right, _ = int(results_right.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * frame_right.shape[1]), int(results_right.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * frame_right.shape[0]), 0

            # draw box around person's nose
            cv2.rectangle(frame_right,(x_right - 30, y_right - 30), (x_right + 30, y_right + 30), (0, 255, 0), 2)

    # calculate depth of person
    if results_left.pose_landmarks is not None and results_right.pose_landmarks is not None:
        disparity = abs(x_left - x_right)
        depth = (focal_length * baseline) / disparity
        cv2.putText(frame_left, f"Depth: {depth:.2f} cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # display left and right images
    cv2.imshow("Left Image", frame_left)
    cv2.imshow("Right Image", frame_right)

    # press 'q' to exit
    if cv2.waitKey(1) == ord('q'): 
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
