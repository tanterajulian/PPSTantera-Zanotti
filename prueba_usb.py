#FUENTE https://forums.developer.nvidia.com/t/cannot-open-opencv-videocapture-with-gstreamer-pipeline/181639

import cv2
cap0 = cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw, format=YUY2 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1", cv2.CAP_GSTREAMER);
#cap0.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
cap0.set(3, 640)
cap0.set(4, 480)

cap1 = cv2.VideoCapture("v4l2src device=/dev/video1 ! video/x-raw, format=YUY2 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1", cv2.CAP_GSTREAMER);
#cap1.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
cap1.set(3, 640)
cap1.set(4, 480)

#cap2 = cv2.VideoCapture(2);
#cap2.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
#cap2.set(3, 640)
#cap2.set(4, 480)

#cap3 = cv2.VideoCapture(3);
#cap3.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
#cap3.set(3, 640)
#cap3.set(4, 480)

while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
  #      ret2, frame2 = cap2.read()
   #     ret3, frame3 = cap3.read()

        print('Retval cap0: ', ret0)
        print('Retval cap1: ', ret1)
    #    print('Retval cap2: ', ret2)
     #   print('Retval cap3: ', ret3)

        if ret0:
                cv2.imshow('frame0', frame0)
        if ret1:
                cv2.imshow('frame1', frame1)
        cv2.waitKey(10)
      #  if ret2:
       #         cv2.imshow('frame2', frame2)
        #if ret3:
         #       cv2.imshow('frame3', frame3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap0.release()
cap1.release()
#cap2.release()
#cap3.release()
cv2.destroyAllWindows()
