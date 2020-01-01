import cv2 as cv
import numpy as np
import time

def connectCamera():
    cap=cv.VideoCapture(2)
    _,frame=cap.read()
    #cap.set(cv.CAP_PROP_FRAME_WIDTH,640)
    #cap.set(cv.CAP_PROP_FRAME_HEIGHT,480)
    cap.set(cv.CAP_PROP_FOCUS,0)
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE,0)
    cap.set(cv.CAP_PROP_EXPOSURE,-7)
    cap.set(cv.CAP_PROP_GAIN,0)
    time.sleep(1)
    _,frame=cap.read()

def dominoesContours(binImg):
    _,contours,hierarchy=cv.findContours(binImg,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    return contours

def contourMask(contour):
    mask=np.zeros(frame.shape,dtype=np.uint8)
    cv.fillPoly(mask,[contour],255)
    return mask

def singleDomino(binImg,mask):
    dst=cv.bitwise_and(binImg,mask)
    return dst

def segmentDomino(binImg,contour):
    dst=binImg.copy()
    leftMost=contour[contour[:,:,0].argmin()][0]
    rightMost=contour[contour[:,:,0].argmax()][0]
    topMost=contour[contour[:,:,1].argmin()][0]
    bottomMost=contour[contour[:,:,1].argmax()][0]
    middlePoints=[]
    middlePoints.append((leftMost+topMost)//2)
    middlePoints.append((leftMost+bottomMost)//2)
    middlePoints.append((rightMost+topMost)//2)
    middlePoints.append((rightMost+bottomMost)//2)
    distance=[]
    distance.append(middlePoints[0]-middlePoints[3])
    distance.append(middlePoints[1]-middlePoints[2])
    distance[0]=np.sum(distance[0]**2)
    distance[1]=np.sum(distance[1]**2)
    if np.argmin(distance)==0:
        cv.line(dst,tuple(middlePoints[0]),tuple(middlePoints[3]),0,5)
    else:
        cv.line(dst,tuple(middlePoints[1]),tuple(middlePoints[2]),0,5)
    return dst

def findNumber(binImg,contour):
    A=cv.countNonZero(binImg)
    if A>2700:
        num=0
    elif A>2500:
        num=1
    else:
        num=2
    M=cv.moments(contour)
    cX=int(M["m10"]/M["m00"])
    cY=int(M["m01"]/M["m00"])
    return [cX,cY,num]

def processImage(binImg):
    ret=[]
    contours=dominoesContours(binImg)
    for contour in contours:
        mask=contourMask(contour)
        tmpImg=singleDomino(binImg,mask)
        tmpImg=segmentDomino(tmpImg,contour)
        
        tmpContours=dominoesContours(tmpImg)
        for tmpContour in tmpContours:
            tmpMask=contourMask(tmpContour)
            ret.append(findNumber(singleDomino(tmpImg,tmpMask),tmpContour))
    return ret
    

frame=cv.imread("frame0.bmp",cv.IMREAD_GRAYSCALE)
_,binar=cv.threshold(frame,80,255,cv.THRESH_BINARY)

print(processImage(binar))
out=frame
for a in processImage(binar):
    cv.putText(out,str(a[2]),(a[0],a[1]),cv.FONT_HERSHEY_COMPLEX,1,0)

#cv.drawContours(color,contours,-1,(0,255,0),3)
#cv.circle(color,(cX,cY),10,(1,227,254),-1)

cv.imwrite('frame'+str(1)+".bmp",out)

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