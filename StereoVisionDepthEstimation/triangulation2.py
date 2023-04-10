import numpy as np
import cv2

def find_depth(right_point, left_point, frame_right, frame_left, baseline, f, alpha):
    # Convert focal length from mm to pixels
    f_px = f / (frame_right.shape[1] / 2)

    # Calculate disparity
    disparity = abs(right_point[0] - left_point[0])

    # Calculate depth
    zDepth = (f_px * baseline) / (disparity + alpha)

    return zDepth

# # Example usage
# # Load left and right frames
# frame_right = cv2.imread('frame_right.png')
# frame_left = cv2.imread('frame_left.png')

# # Define baseline, focal length, and alpha value
# baseline = 0.1 # meters
# f_mm = 10 # millimeters
# alpha = 0 # no additional parameter added to the disparity value

# # Define corresponding points in the right and left frames
# right_point = (100, 200)
# left_point = (70, 200)

# # Calculate depth
# depth = find_depth(right_point, left_point, frame_right, frame_left, baseline, f_mm, alpha)

# print('Depth:', depth, 'meters')
