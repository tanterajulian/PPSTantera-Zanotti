import sys
import cv2
import numpy as np
import time

def add_HSV_filter(frame, camera):
    # Blurring the frame
    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    # Converting RGB to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Set color range for human detection based on camera number
    if camera == 1:
        lower_color = np.array([0, 0, 0])
        upper_color = np.array([30, 255, 255])
    else:
        lower_color = np.array([0, 0, 0])
        upper_color = np.array([30, 255, 255])

    # HSV-filter mask
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Morphological Operation - Opening - Erode followed by Dilate - Remove noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask