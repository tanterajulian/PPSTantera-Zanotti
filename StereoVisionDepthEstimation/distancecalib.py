import cv2
import numpy as np

# Define the size of the chessboard pattern
chessboard_size = (9, 6)

# Create arrays to store the object points and image points for all images
object_points = []
image_points_left = []
image_points_right = []

# Create the object points for the chessboard pattern
object_point = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
object_point[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

# Initialize the camera capture objects
left_camera = cv2.VideoCapture(0)
right_camera = cv2.VideoCapture(1)

right_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
left_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# Set the camera resolution
left_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
left_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
right_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
right_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Capture images of the chessboard pattern from the two cameras
while True:
    ret1, left_image = left_camera.read()
    ret2, right_image = right_camera.read()
    
    if ret1 and ret2:
        # Convert the images to grayscale
        left_image_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_image_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners in the images
        ret_left, corners_left = cv2.findChessboardCorners(left_image_gray, chessboard_size)
        ret_right, corners_right = cv2.findChessboardCorners(right_image_gray, chessboard_size)
        
        if ret_left and ret_right:
            # Draw the chessboard corners on the images
            cv2.drawChessboardCorners(left_image, chessboard_size, corners_left, ret_left)
            cv2.drawChessboardCorners(right_image, chessboard_size, corners_right, ret_right)
            
            # Add the object points and image points to the arrays
            object_points.append(object_point)
            image_points_left.append(corners_left)
            image_points_right.append(corners_right)
            
            # Display the images
            cv2.imshow('Left camera', left_image)
            cv2.imshow('Right camera', right_image)
        
        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Save the calibration parameters to files when the 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Calibrate the cameras using the object points and image points
        left_ret, left_camera_matrix, left_distortion_coefficients, left_rotation_vectors, left_translation_vectors = cv2.calibrateCamera(object_points, image_points_left, (640, 480), None, None)
        right_ret, right_camera_matrix, right_distortion_coefficients, right_rotation_vectors,right_translation_vectors = cv2.calibrateCamera(object_points, image_points_right, (640, 480), None, None)

        #Stereo calibration
        flags = 0
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret, left_camera_matrix, left_distortion_coefficients, right_camera_matrix, right_distortion_coefficients, R, T, E, F = cv2.stereoCalibrate(object_points, image_points_left, image_points_right, left_camera_matrix, left_distortion_coefficients, right_camera_matrix, right_distortion_coefficients, (640, 480), criteria=criteria, flags=flags)

        with open('left_camera_matrix.npy', 'wb') as f:
            np.save(f, left_camera_matrix)
        with open('left_distortion_coefficients.npy', 'wb') as f:
            np.save(f, left_distortion_coefficients)
        with open('right_camera_matrix.npy', 'wb') as f:
            np.save(f, right_camera_matrix)
        with open('right_distortion_coefficients.npy', 'wb') as f:
            np.save(f, right_distortion_coefficients)
        with open('R.npy', 'wb') as f:
            np.save(f, R)
        with open('T.npy', 'wb') as f:
            np.save(f, T)
        print("Calibration parameters saved!")

#Release the camera objects and close the windows

left_camera.release()
right_camera.release()
cv2.destroyAllWindows()


