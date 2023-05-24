import cv2
import numpy as np
import time
import imutils
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
from filterpy.kalman import KalmanFilter

from matplotlib import pyplot as plt

# Function for stereo vision and depth estimation
import triangulation2 as tri
import calibration

# Mediapipe for face detection
import time


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
labels = pd.read_csv('./labels.csv',sep=';',index_col='ID')
labels = labels['OBJECT (2017 REL.)']

dev_L = 0
dev_R = 1
width = 1280
height = 720
# Open both cameras
gst_str_L = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(dev_L, width, height)
gst_str_R = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(dev_R, width, height)

cap_left = cv2.VideoCapture(gst_str_L, cv2.CAP_GSTREAMER)
cap_right = cv2.VideoCapture(gst_str_R, cv2.CAP_GSTREAMER)

# Stereo vision setup parameters
frame_rate = 120    #Camera frame rate (maximum at 120 fps)
B = 9               #Distance between the cameras [cm]
f = 7.8             #Camera lense's focal length [mm]
alpha = 68.6        #Camera field of view in the horisontal plane [degrees]

# Main program loop with face detector and depth estimation using stereo vision
#while(cap_right.isOpened() and cap_left.isOpened()): #! mientras que las camaras estan online
depth_arr = []

# Define the Kalman filter
kf = KalmanFilter(dim_x=1, dim_z=1)
kf.x = np.array([0])  # Initial state estimate
kf.F = np.array([[1]])  # State transition matrix
kf.H = np.array([[1]])  # Measurement matrix
kf.P = np.array([[1]])  # Covariance matrix of initial state estimate
kf.Q = np.array([[0.01]])  # Process noise covariance matrix
kf.R = np.array([[1]])  # Measurement noise covariance matrix

depth2_filtered = []

while(True):

    succes_right, frame_right = cap_right.read() #! lee los cuadros de las camaras
    succes_left, frame_left = cap_left.read()

################## CALIBRATION #########################################################

    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left) #! Rectifica las imagenes

########################################################################################

    # If cannot catch any frame, break
    if not succes_right or not succes_left:             
        break

    else:
        
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
        
        num_detections_r, boxes_r, scores_r, classes_r,_,_,_,_ = detector(frame_right_tensor)
        num_detections_l, boxes_l, scores_l, classes_l,_,_,_,_ = detector(frame_left_tensor)
        
        pred_labels_r = [labels[i] for i in classes_r]
        pred_labels_l = [labels[i] for i in classes_l]

        pred_boxes_r = boxes_r.numpy()[0].astype('int')
        pred_boxes_l = boxes_l.numpy()[0].astype('int')
        
        pred_scores_r = scores_r.numpy()[0]
        pred_scores_l = scores_l.numpy()[0]


        # Convert the RGB image to BGR
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
        

        ################## CALCULATING DEPTH #########################################################


        # if num_detections_r > 0:
        #     for id, detection in enumerate(results_right.detections):
        #         mp_draw.draw_detection(frame_right, detection)

        #         bBox = detection.location_data.relative_bounding_box

        #         h, w, c = frame_right.shape

        #         boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

        #         center_point_right = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

        #         cv2.putText(frame_right, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

        for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores_r, pred_boxes_r, pred_labels_r):

            if score < 0.3:
                continue
            
            score_txt = f'{100 * round(score,0)}'
            if label == "person":
                nobody = 0
                #area_clear = 0
                img_boxes = cv2.rectangle(frame_right,(xmin, ymax),(xmax, ymin),(0,255,0),1)
                center_point_right = (xmax + xmin) / 2 , (ymax + ymin) / 2
                #print("punto ", center_point_right)        
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_boxes,label,(xmin, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
                print(center_point_right)

        for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores_l, pred_boxes_l, pred_labels_l):
            
            if score < 0.3:
                continue
            
            score_txt = f'{100 * round(score,0)}'
            if label == "person":
                nobody = 0
                #area_clear = 0
                img_boxes = cv2.rectangle(frame_left,(xmin, ymax),(xmax, ymin),(0,255,0),1)
                center_point_left = (xmax + xmin) / 2 , (ymax + ymin) / 2
                #print("punto ", center_point_left)      
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_boxes,label,(xmin, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
                cv2.putText(img_boxes,score_txt,(xmax, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
                print(center_point_left)


        # Function to calculate depth of object. 
        if nobody == 0 and center_point_left is not None and center_point_right is not None:

            depth2 = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)


            print("DistanciaSF ", depth2)

            # Update the Kalman filter with the measurement
            kf.predict()
            kf.update(depth2)

            # cv2.putText(frame_right, "Distance: " + str(round(depth2,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            # cv2.putText(frame_left, "Distance: " + str(round(depth2,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
            # print("Depth: ", str(round(depth2,1)))
            depth_arr.append(depth2)

            # Get the filtered estimate of depth2
            depth2_filtered.append(kf.x[0])

            print("DepthKalman: ", kf.x[0])
            
            # Calculo de distancia teniendo en cuenta ultimas 10 mediciones

            if len(depth_arr) > 10:
                depth_arr.pop(0)                #borramos el primer valor que se encuentre en el array
            
            depth_prom = np.mean(depth_arr)     #calculamos el promedio de las mediciones existentes en el array
          
            cv2.putText(frame_right, "Distance: " + str(round(depth_prom,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            cv2.putText(frame_left, "Distance: " + str(round(depth_prom,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)

            print("DepthPromedio:", depth_prom)
        
        else:
            cv2.putText(frame_left, "Area Clear", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            cv2.putText(frame_right, "Area Clear", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            print("no hay nadie")

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
