import cv2
import numpy as np
import time
import imutils
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd

from matplotlib import pyplot as plt

# Function for stereo vision and depth estimation
import triangulation as tri
import calibration

# Mediapipe for face detection
import time


detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
labels = pd.read_csv('labels.csv',sep=';',index_col='ID')
labels = labels['OBJECT (2017 REL.)']

# Open both cameras
cap_right = cv2.VideoCapture(1, cv2.CAP_DSHOW)  #!Inicia camara derecha               
cap_left =  cv2.VideoCapture(0, cv2.CAP_DSHOW)  #!Inicia camara izquierda


# Stereo vision setup parameters
frame_rate = 120    #Camera frame rate (maximum at 120 fps)
B = 9               #Distance between the cameras [cm]
f = 8              #Camera lense's focal length [mm]
alpha = 56.6        #Camera field of view in the horisontal plane [degrees]
h = 640
w = 480



print("APERTURA DE CAMARA 1 ")
print("APERTURA DE CAMARA 0 ")

# Main program loop with face detector and depth estimation using stereo vision
while(True): #! mientras que las camaras estan online

    print("entro while")
    succes_right, frame_right = cap_right.read() #! lee los cuadros de las camaras
    succes_left, frame_left = cap_left.read()

################## CALIBRATION #########################################################

    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left) #! Rectifica las imagenes

########################################################################################

    # If cannot catch any frame, break
    # if not succes_right or not succes_left:                    
    #     break

    # else:

    start = time.time()
    
    # Convert the BGR image to RGB
    frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
    frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

    # Process the image and find faces #!DETECCION

    #Is optional but i recommend (float convertion and convert img to tensor image)
    frame_right_tensor = tf.convert_to_tensor(frame_right, dtype=tf.uint8)
    frame_left_tensor = tf.convert_to_tensor(frame_left, dtype=tf.uint8)

    #Add dims to rgb_tensor
    frame_right_tensor = tf.expand_dims(frame_right_tensor , 0)
    frame_left_tensor = tf.expand_dims(frame_left_tensor , 0)
    
    resp_r = detector(frame_right_tensor)
    resp_l = detector(frame_left_tensor)
    


    # Convert the RGB image to BGR
    frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
    frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)


    ################## CALCULATING DEPTH #########################################################

    center_right = 0
    center_left = 0

    # if num_detections_r > 0:
    #     for id, detection in enumerate(results_right.detections):
    #         mp_draw.draw_detection(frame_right, detection)

    #         bBox = detection.location_data.relative_bounding_box

    #         h, w, c = frame_right.shape

    #         boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

    #         center_point_right = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

    #         cv2.putText(frame_right, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
    
    for boxes, classes, scores in zip(resp_r['detection_boxes'].numpy(), resp_r['detection_classes'], resp_r['detection_scores'].numpy()):
        for box, cls, score in zip(boxes, classes, scores):
            if score > 0.8:
            
                ymin = int(box[0] * h)
                xmin = int(box[1] * w)
                ymax = int(box[2] * h)
                xmax = int(box[3] * w)
                    
                score_txt = f'{100 * round(score,0)}'
                img_boxes = cv2.rectangle(frame_right,(xmin, ymax),(xmax, ymin),(0,255,0),1)
                if cls=="person":
                    center_point_right = (xmax + xmin) / 2 , (ymax + ymin) / 2      
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_boxes,cls,(xmin, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
                cv2.putText(img_boxes,score_txt,(xmax, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)

    for boxes, classes, scores in zip(resp_l['detection_boxes'].numpy(), resp_l['detection_classes'], resp_l['detection_scores'].numpy()):
        for box, cls, score in zip(boxes, classes, scores):
            if score > 0.8:
            
                ymin = int(box[0] * h)
                xmin = int(box[1] * w)
                ymax = int(box[2] * h)
                xmax = int(box[3] * w)
                    
                score_txt = f'{100 * round(score,0)}'
                img_boxes = cv2.rectangle(frame_right,(xmin, ymax),(xmax, ymin),(0,255,0),1)
                if cls=="person":
                    center_point_right = (xmax + xmin) / 2 , (ymax + ymin) / 2      
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_boxes,cls,(xmin, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
                cv2.putText(img_boxes,score_txt,(xmax, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)





        # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
        # All formulas used to find depth is in video presentaion
    depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)

    cv2.putText(frame_right, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
    cv2.putText(frame_left, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
    # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
    print("Depth: ", str(round(depth,1)))



    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
    #print("FPS: ", fps)

    cv2.putText(frame_right, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.putText(frame_left, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)                                   


    # Show the frames
    cv2.imshow("frame right", frame_right) 
    cv2.imshow("frame left", frame_left)


    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()
