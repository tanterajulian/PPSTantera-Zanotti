import sys
import cv2
import numpy as np
import time
import imutils

# Load the pre-trained HOG descriptor and SVM classifier
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def find_person(frame):
    center = None

    # Detect people in the frame using the HOG descriptor and SVM classifier
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
        padding=(8, 8), scale=1.05)

    # Draw a rectangle around each detected person and find the center
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        center = (int(x + w/2), int(y + h/2))

    return center