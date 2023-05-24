import cv2
import numpy as np
from filterpy.kalman import KalmanFilter


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
frame_rate = 120    #Camera frame rate (maximum at 120 fps)
B = 20             #Distance between the cameras [cm]
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


        # if num_detections_r > 0:
        #     for id, detection in enumerate(results_right.detections):
        #         mp_draw.draw_detection(frame_right, detection)

        #         bBox = detection.location_data.relative_bounding_box

        #         h, w, c = frame_right.shape

        #         boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)

        #         center_point_right = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)

        #         cv2.putText(frame_right, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

        center_point_right = 0
        center_point_left = 0

        for detection in detections_right:
            center_point_right = detection.Center

        for detection in detections_left:
            center_point_left = detection.Center


        # Function to calculate depth of object. 
        if center_point_left != 0 and center_point_right != 0:
 
            depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)

            depth_cm = depth*100

            # Update the Kalman filter with the measurement
            kf.predict()
            kf.update(depth_cm)

            # cv2.putText(frame_right, "Distance: " + str(round(depth2,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            # cv2.putText(frame_left, "Distance: " + str(round(depth2,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
            # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
            # print("Depth: ", str(round(depth2,1)))
            depth_arr.append(depth_cm)

            # Get the filtered estimate of depth2
            depth2_filtered.append(kf.x[0])

            print("DepthKalman: ", kf.x[0])
            
            # Calculo de distancia teniendo en cuenta ultimas 10 mediciones

            if len(depth_arr) > 10:
                depth_arr.pop(0)                #borramos el primer valor que se encuentre en el array
            
            depth_prom = np.mean(depth_arr)     #calculamos el promedio de las mediciones existentes en el array

            # Ejemplo de uso
            real_distance = estimated_to_real(depth_prom)
            # real_distance = depth_prom
            print("Distancia real estimada para una medida de", depth_prom, "es:", real_distance, "cm")

            print("DepthPromedio:", real_distance)
        
        else:
            print("no hay nadie")

        output0.Render(img0)
        output0.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))
        output1.Render(img1)
        output1.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))



cv2.destroyAllWindows()
