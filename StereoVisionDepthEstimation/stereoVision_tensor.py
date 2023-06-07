import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
import datetime


# Function for stereo vision and depth estimation
import triangulation2 as tri
import calibration


from jetson.inference import detectNet
from jetson.utils import videoOutput, videoSource, cudaAllocMapped, cudaConvertColor, cudaDeviceSynchronize, cudaToNumpy

import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

input0 = videoSource('/dev/video0', argv=sys.argv)
input1 = videoSource('/dev/video1', argv=sys.argv)
output0 = videoOutput(args.output, argv=sys.argv+is_headless)
output1 = videoOutput(args.output, argv=sys.argv+is_headless)

net = detectNet(args.network, sys.argv, args.threshold)


# Stereo vision setup parameters
frame_rate = 30    #Camera frame rate (maximum at 120 fps)
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

# Datos de medidas reales y estimadas
real_distances = np.array([100, 200, 300, 400, 500])
estimated_distances = np.array([185, 299, 361, 420, 450])

# Ajuste de polinomio
polyfit_coeffs = np.polyfit(estimated_distances, real_distances, deg=4)

# Función polinómica para estimar distancia real a partir de la medida estimada
def estimated_to_real(estimated_distance):
    return np.polyval(polyfit_coeffs, estimated_distance)

while(True):
    img0 = input0.Capture()
    img1 = input1.Capture()

    # convert to BGR, since that's what OpenCV expects
    bgr_img_0 = cudaAllocMapped(width=img0.width,
                            height=img0.height,
                            format='bgr8')
    bgr_img_1 = cudaAllocMapped(width=img1.width,
                            height=img1.height,
                            format='bgr8')

    cudaConvertColor(img0, bgr_img_0)
    cudaConvertColor(img1, bgr_img_1)

    # make sure the GPU is done work before we convert to cv2
    cudaDeviceSynchronize()

    # convert to cv2 image (cv2 images are numpy arrays)
    frame_right = cudaToNumpy(bgr_img_0)
    frame_left = cudaToNumpy(bgr_img_1)


################## CALIBRATION #########################################################

    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left) #! Rectifica las imagenes

########################################################################################

    # If cannot catch any frame, break
    if not input0.IsStreaming() or not input1.IsStreaming():             
        break

    else:
        
        detections_right = net.Detect(img0, overlay=args.overlay)
        detections_left = net.Detect(img1, overlay=args.overlay)

        

        ################## CALCULATING DEPTH #########################################################

        center_point_right = 0
        center_point_left = 0

        for (score_r, (ymin_r, xmin_r, ymax_r, xmax_r), label_r), (score_l, (ymin_l, xmin_l, ymax_l, xmax_l), label_l) in zip(zip(pred_scores_r, pred_boxes_r, pred_labels_r), zip(pred_scores_l, pred_boxes_l, pred_labels_l)):
            
            score_txt = f'{100 * round(score_r,0)}'
            if label_r == "person" and score_r >= 0.3:
                nobody = 0
                img_boxes = cv2.rectangle(frame_right, (xmin_r, ymax_r), (xmax_r, ymin_r), (0, 255, 0), 1)
                center_point_right = ((xmax_r + xmin_r) / 2, (ymax_r + ymin_r) / 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_boxes, label_r, (xmin_r, ymax_r - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            
            score_txt = f'{100 * round(score_l,0)}'
            if label_l == "person" and score_l >= 0.3:
                nobody = 0
                img_boxes = cv2.rectangle(frame_left, (xmin_l, ymax_l), (xmax_l, ymin_l), (0, 255, 0), 1)
                center_point_left = ((xmax_l + xmin_l) / 2, (ymax_l + ymin_l) / 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img_boxes, label_l, (xmin_l, ymax_l - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(img_boxes, score_txt, (xmax_l, ymax_l - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)


        # Function to calculate depth of object. 
        if center_point_left != 0 and center_point_right != 0:
 
            depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)

            depth_cm = depth*100

            # Update the Kalman filter with the measurement
            kf.predict()
            kf.update(depth_cm)

            depth_arr.append(depth_cm)

            # Get the filtered estimate of depth2
            depth2_filtered.append(kf.x[0])

            print("DepthKalman: ", kf.x[0])

            #Si la distancia medida es menor a 200, imprimimos en un archivo log 

            if kf.x[0] < 200:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open('/home/julian/PPSTantera-Zanotti/StereoVisionDepthEstimation/logDomo.txt', 'w') as file:
                    file.write(f"Time: {current_time}, Distancia a la persona: {kf.x[0]}\n")

                # Capture images from each camera
                ret, left_image = cap_left.read()
                ret, right_image = cap_right.read()

                # Save the images (overwrite the existing images)
                left_image_path = "/home/julian/PPSTantera-Zanotti/StereoVisionDepthEstimation/left_image.jpg"
                right_image_path = "/home/julian/PPSTantera-Zanotti/StereoVisionDepthEstimation/right_image.jpg"

                cv2.imwrite(left_image_path, left_image)
                cv2.imwrite(right_image_path, right_image)

            
            # Calculo de distancia teniendo en cuenta ultimas 10 mediciones

            # if len(depth_arr) > 10:
            #     depth_arr.pop(0)                #borramos el primer valor que se encuentre en el array
            
            # depth_prom = np.mean(depth_arr)     #calculamos el promedio de las mediciones existentes en el array
          
            # cv2.putText(frame_right, "Distance: " + str(round(depth_prom,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            # cv2.putText(frame_left, "Distance: " + str(round(depth_prom,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)

            # print("DepthPromedio:", depth_prom)
        
        else:
            print("no hay nadie")

        output0.Render(img0)
        output0.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))
        output1.Render(img1)
        output1.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))



cv2.destroyAllWindows()
