import cv2 as cv
import numpy as np
import time

cap=cv.VideoCapture(2)
#cap.set(cv.CAP_PROP_FRAME_WIDTH,640)
#cap.set(cv.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv.CAP_PROP_FOCUS,0)
cap.set(cv.CAP_PROP_AUTO_EXPOSURE,0)
cap.set(cv.CAP_PROP_EXPOSURE,-6)
cap.set(cv.CAP_PROP_GAIN,0)
time.sleep(1)
ret,frame=cap.read()
cv.imwrite('frame'+str(0)+".bmp",frame)
while True:
#for i in range(5):
    ret,frame=cap.read()
    #cv.imwrite('frame'+str(i)+".bmp",frame)
    #if frame.shape[0] < frame.shape[1]:
    #    frame = cv.resize(frame, (266, 200))
    #else:
    #    frame = cv.resize(frame, (200, 266))
    cv.imshow("A",frame)
    print(cap.get(cv.CAP_PROP_FOCUS),cap.get(cv.CAP_PROP_BRIGHTNESS),cap.get(cv.CAP_PROP_GAIN))
    if cv.waitKey(33)==ord('q'):
        break
cap.release()
cv.destroyAllWindows() 