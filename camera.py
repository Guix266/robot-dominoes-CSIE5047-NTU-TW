import cv2 as cv
import numpy as np
import time
import math

def calcScale(p1_img,p1_arm,p2_img,p2_arm,p3_img,p3_arm):
    a = np.array([[p1_img[0], p1_img[1],0,0, 1,0],
                  [0,0,p1_img[0], p1_img[1], 0,1],
                  [p2_img[0], p2_img[1],0,0, 1,0],
                  [0,0,p2_img[0], p2_img[1], 0,1], 
                  [p3_img[0], p3_img[1],0,0, 1,0],
                  [0,0,p3_img[0], p3_img[1], 0,1]])
    b= np.array([p1_arm[0],
                 p1_arm[1],
                 p2_arm[0],
                 p2_arm[1], 
                 p3_arm[0],
                 p3_arm[1]])
    x = np.linalg.solve(a, b)
    print(x)
    return x

def calcCentroid(p1,p2):
    print((p1[0]+p2[0])/2,(p1[1]+p2[1])/2)

def calcTest(p):
    x=-0.0099*p[0]-0.4007*p[1]+376.442
    y=-0.4245*p[0]-0.0136*p[1]+149.549
    print(x,y)

def dominoesContours(binImg):
    _,contours,hierarchy=cv.findContours(binImg,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    return contours

def contourMask(contour):
    mask=np.zeros((480,640),dtype=np.uint8)
    cv.fillPoly(mask,[contour],255)
    return mask

def singleDomino(binImg,mask):
    dst=cv.bitwise_and(binImg,mask)
    return dst
'''
def segmentDomino2(binImg,contour):
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
        cv.circle(dst,tuple(middlePoints[0]),5,0,-1)
        cv.circle(dst,tuple(middlePoints[3]),5,0,-1)
    else:
        cv.line(dst,tuple(middlePoints[1]),tuple(middlePoints[2]),0,5)
        cv.circle(dst,tuple(middlePoints[1]),5,0,-1)
        cv.circle(dst,tuple(middlePoints[2]),5,0,-1)
    return dst
'''
def segmentDomino(binImg,contour):
    dst=binImg.copy()
    M=cv.moments(contour)
    cX=int(M["m10"]/M["m00"])
    cY=int(M["m01"]/M["m00"])
    minDistance=99999
    minX=0
    minY=0
    for p in contour:
        d=(p[0][0]-cX)**2+(p[0][1]-cY)**2
        if d<minDistance:
            minDistance=d
            minX=p[0][0]
            minY=p[0][1]
    cv.line(dst,(minX,minY),(2*cX-minX,2*cY-minY),0,5)
    ##cv.imwrite('frame'+str(1)+".bmp",dst)
    return dst

def findNumber(binImg,contour):
    A=cv.countNonZero(binImg)
    if A>2800:
        num=0
    elif A<2400:
        num=2
    else:
        num=1
    M=cv.moments(contour)
    cX=int(M["m10"]/M["m00"])
    cY=int(M["m01"]/M["m00"])
    print(str([cX,cY,num])+" has "+str(A)+" pixels")
    return [cX,cY,num]

def processImage(binImg):
    ret=[]
    contours=dominoesContours(binImg)
    print("The number of domino contours:"+str(len(contours)))
    for contour in contours:
        mask=contourMask(contour)
        print("Domino "+" mask has "+str(cv.countNonZero(mask))+" pixels")
        if cv.countNonZero(mask)<3000:
            print("Skip: the number of domino mask pixels is "+str(cv.countNonZero(mask)))
            continue
        tmpImg=singleDomino(binImg,mask)
        tmpImg=segmentDomino(tmpImg,contour)
        cv.imwrite('frame'+str(1)+".bmp",tmpImg)
        tmpContours=dominoesContours(tmpImg)
        print("The number of square contours:"+str(len(tmpContours)))
        for tmpContour in tmpContours:
            tmpMask=contourMask(tmpContour)
            print("Square "+" mask has "+str(cv.countNonZero(tmpMask))+" pixels")
            if cv.countNonZero(tmpMask)<1000:
                print("Skip: the number of square mask pixels is "+str(cv.countNonZero(tmpMask)))
                continue
            ret.append(findNumber(singleDomino(tmpImg,tmpMask),tmpContour))
    return ret

def dominoAngle(input):
    angle=[]
    y=len(input)
    for i in range(0,y,2):
        first=input[i]
        second=input[i+1]
        first_number=first[2]
        second_number=second[2]
        if first_number <= second_number:
            angle.append(math.degrees(-math.atan2(second[1]-first[1],second[0]-first[0])))
        else:
            angle.append(math.degrees(-math.atan2(first[1]-second[1],first[0]-second[0])))
    return angle

def finalOutput(img):
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    _,binar=cv.threshold(img,40,255,cv.THRESH_BINARY)
    ret=[]
    processed=processImage(binar)
    angle=dominoAngle(processed)
    for i in range(len(angle)):
        x=(processed[i*2][0]+processed[i*2+1][0])//2
        y=(processed[i*2][1]+processed[i*2+1][1])//2
        ret.append([str(processed[i*2][2])+str(processed[i*2+1][2]),(x,y,angle[i])])
    return ret
    
'''    
contours=dominoesContours(binar)

mask=contourMask(contours[2])
tmpImg=singleDomino(binar,mask)
tmpImg=segmentDomino(tmpImg,contours[2])

tmpContours=dominoesContours(tmpImg)
tmpMask=contourMask(tmpContours[1])
findNumber(singleDomino(tmpImg,tmpMask),tmpContours[1])
tmpImg=singleDomino(tmpImg,tmpMask)
cv.countNonZero(tmpImg)
'''



'''
color=tmpImg
color=cv.cvtColor(tmpImg,cv.COLOR_GRAY2BGR)
cv.drawContours(color,tmpContours,-1,(0,255,0),3)
'''
#cv.circle(color,(cX,cY),10,(1,227,254),-1)

#print(dominoAngle(processImage(binar)))


'''
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
'''