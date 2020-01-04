# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 23:54:47 2020

@author: Sora
"""
import numpy as np
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
    
    
    
    #c = np.array([[p1_img[0], p1_img[1], 1], [p2_img[0], p2_img[1], 1], [p3_img[0], p3_img[1], 1]])
    #d= np.array([p1_arm[1], p2_arm[1], p3_arm[1]])
    #y = np.linalg.solve(c, d)
    #print(y)
    
    #print("armX="+str(x[0])+"imgX+"+str(x[1])+"imgY+"+str(x[2]))
    #print("armY="+str(y[0])+"imgX+"+str(y[1])+"imgY+"+str(y[2]))
    
    
if __name__ == '__main__':
    p1_img=[241.5,179.5]
    p2_img=[71.5,156.0]
    p3_img=[540.5,371.5]
    p1_arm=[302.1,34.6]
    p2_arm=[307.2,108.2]
    p3_arm=[219.2,-95.3]

    answer=calcScale(p1_img,p1_arm,p2_img,p2_arm,p3_img,p3_arm)
    ax=332.5*answer[0]+173.0*answer[1]+answer[4]
    ay=332.5*answer[2]+173.0*answer[3]+answer[5]