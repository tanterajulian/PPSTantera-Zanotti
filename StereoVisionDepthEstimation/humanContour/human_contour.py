import sys
import cv2
import numpy as np
import time
import imutils

def find_person(frame, mask):

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # Only proceed if at least one contour was found
    if len(contours) > 0:
        # Find the largest contour in the mask, then use
        # it to compute the minimum enclosing rectangle and
        # centroid
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        M = cv2.moments(c)       #Finds center point
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Only proceed if the width and height are greater than a minimum value
        if w > 50 and h > 50:
            # Draw the rectangle and centroid on the frame,
            # then update the list of tracked points
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 0), -1)

    return center
