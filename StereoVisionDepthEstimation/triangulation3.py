import cv2
import numpy as np

def find_depth(right_point, left_point, frame_right, frame_left, baseline, f, alpha):
    # Convert focal length from mm to pixels using width of right frame
    focal_length_pixels = f * frame_right.shape[1] / 36

    # Calculate disparity between right and left points
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(frame_left, frame_right)

    # Calculate depth using triangulation
    # x = (u - cx) / fx
    # z = baseline * fx / disparity
    u_r, v_r = right_point
    u_l, v_l = left_point
    x_r = (u_r - right_camera_matrix[0, 2]) / right_camera_matrix[0, 0]
    x_l = (u_l - left_camera_matrix[0, 2]) / left_camera_matrix[0, 0]
    depth = baseline * focal_length_pixels / (disparity[v_r, u_r] - alpha * (x_r - x_l))

    return depth