import cv2
import numpy as np

# Load left and right camera calibration data
left_camera_matrix = np.load('left_camera_matrix.npy')
left_distortion_coefficients = np.load('left_distortion_coefficients.npy')
right_camera_matrix = np.load('right_camera_matrix.npy')
right_distortion_coefficients = np.load('right_distortion_coefficients.npy')
R = np.load('R.npy')
T = np.load('T.npy')

# Set up left and right cameras
left_camera = cv2.VideoCapture(0)
right_camera = cv2.VideoCapture(1)

while True:
    # Capture frames from left and right cameras
    ret1, left_frame = left_camera.read()
    ret2, right_frame = right_camera.read()

    # Undistort frames using calibration data
    left_frame_undistorted = cv2.undistort(left_frame, left_camera_matrix, left_distortion_coefficients)
    right_frame_undistorted = cv2.undistort(right_frame, right_camera_matrix, right_distortion_coefficients)

    # Rectify frames using stereo calibration data
    R1, R2, P1, P2, Q, _ = cv2.stereoRectify(left_camera_matrix, left_distortion_coefficients, right_camera_matrix, right_distortion_coefficients, (left_frame_undistorted.shape[1], left_frame_undistorted.shape[0]), R, T, flags=cv2.CALIB_ZERO_DISPARITY)
    left_frame_rectified = cv2.remap(left_frame_undistorted, R1, P1, cv2.INTER_LINEAR)
    right_frame_rectified = cv2.remap(right_frame_undistorted, R2, P2, cv2.INTER_LINEAR)

    # Compute disparity map
    stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=5)
    disparity_map = stereo.compute(left_frame_rectified, right_frame_rectified)

    # Compute depth map using triangulation
    depth_map = cv2.reprojectImageTo3D(disparity_map, Q)

    # Display depth map
    cv2.imshow('Depth Map', depth_map)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release cameras and close window
left_camera.release()
right_camera.release()
cv2.destroyAllWindows()
