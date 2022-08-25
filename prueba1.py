import cv2
cap0 = cv2.VideoCapture(0);
cap0.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
        ret0, frame1 = cap0.read()

        print('Retval cap0: ' ,ret0)

        if ret0:
                cv2.imshow('frame1', frame1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap0.release()
cv2.destroyAllWindows()
