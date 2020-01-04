# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:52:01 2019

@author: guix
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2 as cv
import os
import time

# import the modules that we have created
import AI_v2
from arm import *
from camera import *

def coord_board(X, Y, angle):
    """ x, y : int
            Pixel coordinates of the center of the domino
        phi : float
            The angle of a vector pointing from the half with lower number of points to the half with higher number.
            Is defined with respect to the y-axis counterclockwise.
    """
    bl_corner = (372.2957, 147.126)
    scale = 0.41408
    X_robot = bl_corner[0] - scale*Y
    Y_robot = bl_corner[1] - scale*X
    angle = angle - 90
    return(X_robot, Y_robot, angle)

def coord_hand(X, Y, angle):
    """ x, y : int
            Pixel coordinates of the center of the domino
        phi : float
            The angle of a vector pointing from the half with lower number of points to the half with higher number.
            Is defined with respect to the y-axis counterclockwise.
    """
    bl_corner = (150,-200)
    scale = 0.41408
    X_robot = bl_corner[0] - scale*X
    Y_robot = bl_corner[1] + scale*Y
    angle = angle - 180
    return(X_robot, Y_robot, angle)
    



# =============================================================================
# Define the game
# =============================================================================

# Will contain the dominoes objects on the board
Board = []

# Number of tilt in each hand
m = 5


# =============================================================================
# 1rst round
# =============================================================================

# =============================================================================
# # I) The robot go on top of the board and recognise the first domino
# =============================================================================

# initialization
# myDobot=arm.DobotDominoes()
# myDobot.arm.setHome(200, 0, 170, 0) # x, y, z, theta
cap=cv.VideoCapture(0)
_,frame=cap.read()
cap.set(cv.CAP_PROP_FOCUS,0)
cap.set(cv.CAP_PROP_AUTO_EXPOSURE,0)
cap.set(cv.CAP_PROP_EXPOSURE,-7)
cap.set(cv.CAP_PROP_GAIN,0)
time.sleep(1)
_,frame=cap.read()

# =============================================================================
# # II) Recognition of the first Board
# =============================================================================

#*** Camera take a picture
_,frame=cap.read()
#for the test import an image
'''
folder = "./"
filename = "frame2.bmp"
img = cv.imread(os.path.join(folder, filename), 1)
if img.shape[0] < img.shape[1]:
    img = cv.resize(img, (266, 200))
else:
    img = cv.resize(img, (200, 266))

plt.imshow(img)

# Recognition of the first domino
result = image_recognition.find_dominoes(img, 12)[0]
'''
result=finalOutput(frame)
# get the spacial informations of the domino annd convert them into real coordinates
domino = result[0]
start_X, start_Y, start_angle = coord_board(domino[1][0],domino[1][1],domino[1][2])

# Add the domino to the list of dominoes on the board
dom = AI_v2.Starting_Domino(domino[0], start_X, start_Y, start_angle) # ("11", 654, 76, 
Board.append(dom)

# =============================================================================
# # III) Robot go on top of his hand
# =============================================================================

#***
# myDobot.arm.goSuck(200,0)


# =============================================================================
# # IV) Robot choose the adapted tilt
# =============================================================================

#*** Camera take a picture
#for the test import an image
'''
folder = "frames"
filename = "frame1.bmp"
img = cv.imread(os.path.join(folder, filename), 1)
if img.shape[0] < img.shape[1]:
    img = cv.resize(img, (266, 200))
else:
    img = cv.resize(img, (200, 266))


result = image_recognition.find_dominoes(img, 11) #! m
'''
_,frame=cap.read()
result=finalOutput(frame)
#print(result)
robot_hand = []
for domino in result:
    robot_hand.append(domino[0])
# print(robot_hand)

# Compute all the possible plays of robot
parent_free_on_board, possibles = AI_v2.show_possibilities(robot_hand, Board)
#Choose the most adapted play
play = AI_v2.better_play(possibles) # play = ["11", dom_parent, num_connection]
print(play)

# Add to the list this dominoes
if play == False:
    print("No play available for the robot. ")
    input("Please draw for the robot and type [ENTER]")
else :
    print("The robot plays [ "+str(play[0][0])+" | "+str(play[0][1])+" ] on "+str(play[1]))
    
    dom = AI_v2.play_this_domino(play[0], play[1])
    Board.append(dom)
    
# Get the position of the domino in the hand
for domino in result:
    if domino[0] == play[0]:
        play_real = domino      # play_hand = ["11", (X_hand, Y_hand, angle_hand), phi]

# =============================================================================
# # V) Robot need to pick the tilt
# =============================================================================

# convert the coordinates in """"play_hand"""" to robot coordinates
play_real[1] = coord_hand(play_real[1][0],play_real[1][1],play_real[1][2])
    
#*** pick the tilt to the good coordinates
myDobot.arm.goSuck(play_real[1][0], play_real[1][1])

# =============================================================================
# # VI) Robot go back in front of the board and place the tilt
# =============================================================================

#*** put the tilt to the good coordinates """"contained in dom.x, dom.y, dom.angle""""
myDobot.arm.goDisSuck(dom.x, dom.y)
# ATTENTION NEED TO RESCALE THE ANGLE BETWEEN -180,180
 
myDobot.arm.rotateAbs(dom.angle - play_real[1][2])


# =============================================================================
# # VII) Robot goes back in the top of the board
# =============================================================================

#***
            
# =============================================================================
# # VIII) Wait for the player to play
# =============================================================================
        
input("It is your turn, please play and type ENTER...")


# =============================================================================
# # IX) Recognition of the board
# =============================================================================

#*** We just need to addapt what we have done before and make a loop

    
_,frame=cap.read()


print(processImage(binar))
out=frame
for a in processImage(binar):
    cv.circle(out,(a[0],a[1]),5,255,-1)
    cv.putText(out,str(a[2]),(a[0],a[1]),cv.FONT_HERSHEY_COMPLEX,1,255)

print(finalOutput(binar))
cv.imwrite('frame'+str(1)+".bmp",out)