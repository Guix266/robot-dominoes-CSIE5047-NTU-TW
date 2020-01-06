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
import AI_v3
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

    # Need to be changed during the calibration !!!!!!!!!!!!!!!!!!!!!
    # take a point in front of the camera + take coordinate pixel on the image + take coordinates on the robot frame
    '''
    pixel_coord = (178,168)
    real_coord = (303.5, 66.5)
    scale = 0.41408
    '''
    X_robot = boardMat[0]*X+boardMat[1]*Y+boardMat[4]
    Y_robot = boardMat[2]*X+boardMat[3]*Y+boardMat[5]
    angle = angle
    return(X_robot, Y_robot, angle)


def coord_hand(X, Y, angle):
    """ x, y : int
            Pixel coordinates of the center of the domino
        phi : float
            The angle of a vector pointing from the half with lower number of points to the half with higher number.
            Is defined with respect to the y-axis counterclockwise.
    """

    bl_corner = (150, -200)
    '''
    bl_corner = (150,-200)
    scale = 0.41408
    '''
    X_robot = handMat[0]*X+handMat[1]*Y+handMat[4]
    Y_robot = handMat[2]*X+handMat[3]*Y+handMat[5]
    angle = angle - 90
    return(X_robot, Y_robot, angle)
    

def principal_angle(angle):
    while angle >= 180:
        angle -= 360
    while angle <= -180:
        angle += 360
    return(angle)


def real_rotate(angle, thres):
    """
    thres: max angle of rotation of the suction cup when the robot is 
    in the _neutral position_
    """
    if abs(angle) <= thres:
        index = 0
    elif angle > thres:
        myDobot.rotateRel(-angle + thres)
        index = 1
    elif angle < -thres:
        myDobot.rotateRel(-angle - thres)
        index = 2
    return index

def distance(dom1, dom2):
    """ give the distnace between 2 centers
    input : ["11", (X,Y,angle)]  """
    distance = (dom1[1][0] - dom2[1][0])**2 + (dom1[1][1] - dom2[1][1])**2
    return((distance)**(1/2))

# =============================================================================
# Define the game
# =============================================================================

# Will contain the dominoes objects on the board
Board = []

# Number of tilt in each hand
m = 3

# calibration
boardMat=calcScale((564,336),(238.7,-97.7),(336,252),(271.1,-0.9),(169.5,236),(277.4,67.4))
handMat=calcScale((247.5,190.5),(36.1,-300),(377,204.5),(-19,-295.9),(521.5,203.5),(-79.8,-296))
#boardMat=calcScale((241.5,179.5),(302.1,34.6),(71.5,156),(307.2,108.2),(540.5,371.5),(219.2,-95.3))
#handMat=calcScale((384.5,144),(-31.9,-317.1),(407,319.5),(-36.6,-242.4),(535.5,214),(-93.4,-288.9))
# =============================================================================
# # I) The robot go to the top of the board and recognise the first domino
# =============================================================================

# initialization
myDobot=DobotDominoes()
time.sleep(1)
myDobot.setHome(200,0,170,0) # x, y, z, theta
myDobot.goHome()

cap=cv.VideoCapture(0)
_,frame=cap.read()
cap.set(cv.CAP_PROP_FOCUS, 0)
cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv.CAP_PROP_EXPOSURE, -6)
cap.set(cv.CAP_PROP_GAIN, 0)
time.sleep(2)
time.sleep(1)
input("blaa")
_,frame = cap.read()
_,frame = cap.read()
time.sleep(1)
plt.imshow(frame)


# =============================================================================
# # II) Recognition of the first Board
# =============================================================================

'''
# Recognition of the first domino
result = image_recognition.find_dominoes(img, 12)[0]
'''

# get the spacial informations of the domino and convert them into real coordinates
result=finalOutput(frame)

#calcScale((527.5,161),(306.7,-76.6),(226,147),(315.3,51.6),(299,258),(270.1,19.1))
# get the spacial information of the domino and convert them into real coordinates

# domino = result[0]
domino = ["11", result[0][1]]
start_X, start_Y, start_angle = coord_board(domino[1][0], domino[1][1], domino[1][2])

# Add the domino to the list of dominoes on the board
dom = AI_v3.Starting_Domino(domino[0], start_X, start_Y, start_angle) # ("11", 654, 76, 
Board.append(dom)

# =============================================================================
# # III) Robot go on top of his hand
# =============================================================================

# myDobot.arm.goSuck(200,0)
myDobot.goTopHand()
time.sleep(2)

# =============================================================================
# # IV) Robot choose the adapted tilt
# =============================================================================

# Camera take a picture
_,frame = cap.read()
_,frame = cap.read()
time.sleep(1)

result_hand = finalOutput(frame)
#input("blaa")
#print(result_hand)
robot_hand = []
for domino in result_hand:
    robot_hand.append(domino[0])
# print(robot_hand)

# Compute all the possible plays of robot
parent_free_on_board, possibles = AI_v3.show_possibilities(robot_hand, Board)
# Choose the most adapted play
play = AI_v3.better_play(possibles) # play = ["11", dom_parent, num_connection]
print(play)

# Add to the list this dominoes
if not play:
    print("No move available for the robot. ")
    input("Please draw for the robot and type [ENTER]")
else:
    print("The robot plays [ "+str(play[0][0])+" | "+str(play[0][1])+" ] on "+str(play[1]))
    
    dom = AI_v3.play_this_domino(play[0], play[1])
    Board.append(dom)
    
# Get the position of the domino in the hand
for domino in result_hand:
    if domino[0] == play[0]:
        play_real = domino      # play_hand = ["11", (X_hand, Y_hand, angle_hand), phi]

# =============================================================================
# # V) Robot need to pick the tilt
# =============================================================================

# convert the coordinates in """"play_hand"""" to robot coordinates
play_real[1] = coord_hand(play_real[1][0],play_real[1][1],play_real[1][2])


# prepare for the rotation
'''
angle = principal_angle(dom.angle - play_real[1][2])
index = real_rotate(angle)
'''

# if the domino has to be rotated by a very large angle, rotate the suction cup first 
rotate_in_air = principal_angle(dom.angle - play_real[1][2]) - 15
real_rotate(rotate_in_air, 140)
time.sleep(2)
# pick the tilt to the good coordinates
print(play_real[1][0], play_real[1][1])
myDobot.goSuck(play_real[1][0], play_real[1][1])

# =============================================================================
# # VI) Robot go back in front of the board and place the tilt
# =============================================================================

myDobot.goTop()
myDobot.rotateRel(rotate_in_air)
myDobot.goDisSuck(dom.x, dom.y)
time.sleep(2)
myDobot.goTop()
myDobot.rotateRel(-rotate_in_air)
time.sleep(2)
            
# =============================================================================
# # VIII) Wait for the player to play
# =============================================================================
        
input("It is your turn, please play and type ENTER...")


# =============================================================================
# # IX) Recognition of the board
# =============================================================================

#*** We just need to addapt what we have done before and make a loop
while(True):
    
    myDobot.rotateAbs(0)
    
    # =============================================================================
    # # VII) Robot goes back in the top of the board and learn which domino was added by the player
    # =============================================================================
    
    # compute dominoes on the board before picture
    board_before = []
    for domino in Board:
        board_before.append(domino.name)
        
    myDobot.goTop()
    time.sleep(2)
    _,frame = cap.read()
    _,frame = cap.read()
    time.sleep(1)
    result = finalOutput(frame)
    #input("blaa")

    # find the new domino
    new_domino = []
    for i in range(len(result)):
        if result[i][0] not in board_before:
            new_domino = result[i]
    
    # get the parent of new_domino
    if distance(new_domino, result[0]) > 1e-5:
        new_parent = result[0]
    else:
        new_parent = result[1]
    min_dist = distance(new_parent, new_domino)
    for i in range(0, len(result)):
        if distance(new_domino, result[i]) < min_dist and distance(new_domino, result[i])>0 :
            new_parent = result[i]
            min_dist = distance(new_parent, new_domino)
    
    if new_domino == None:
        raise Exception("I can't find a new domino on the board!")
    print("I see a new domino: %s" % str(new_domino))
    print("On this parent %s" % str(new_parent))
    
    for obj in Board:
        print(obj.name)
        if obj.name == new_parent[0]:
            new_parent = obj
            break
    if type(new_parent) == list:
        new_parent = Board[-1]
    # Add the new_domino to the list of dominoes on the board
    dom = AI_v3.play_this_domino(new_domino[0], new_parent)
    Board.append(dom)
    
    # =============================================================================
    # # III) Robot go on top of his hand
    # =============================================================================

    time.sleep(5)
    myDobot.goTopHand()
    time.sleep(5)
    _,frame=cap.read()
    _,frame = cap.read()
    time.sleep(1)
    #cv.imwrite('frame'+str(1)+".bmp",frame)
    result_hand = finalOutput(frame)
    #input("blaa")
    robot_hand = []
    for domino in result_hand:
        robot_hand.append(domino[0]) # name of the domino
    print("Robot hand: " + str(robot_hand))
    
    # Compute all the possible plays of robot
    parent_free_on_board, possibles = AI_v3.show_possibilities(robot_hand, Board)
    #Choose the most adapted play
    play = AI_v3.better_play(possibles) # play = ["11", dom_parent, num_connection]
    print(play)
    
    if play == False:
        print("No play available for the robot. ")
        input("Please draw for the robot and type [ENTER]")
    else :
        print("The robot plays [ "+str(play[0][0])+" | "+str(play[0][1])+" ] on "+str(play[1]))
        
        dom = AI_v3.play_this_domino(play[0], play[1])
        Board.append(dom)
        
    # Get the position of the domino in the hand
    for domino in result_hand:
        if domino[0] == play[0]:
            play_real = domino      # play_hand = ["11", (X_hand, Y_hand, angle_hand), phi]
    
    play_real[1] = coord_hand(play_real[1][0],play_real[1][1],play_real[1][2])
    
    # if the domino has to be rotated by a very large angle, rotate the suction cup first 
    rotate_in_air = principal_angle(dom.angle - play_real[1][2]) - 15
    real_rotate(rotate_in_air, 140)
    time.sleep(2)
    
    # pick the tilt to the good coordinates
    myDobot.goSuck(play_real[1][0], play_real[1][1])

    # =============================================================================
    # # VI) Robot go back in front of the board and place the tilt
    # =============================================================================
    
    myDobot.goTop()
    myDobot.rotateRel(rotate_in_air)
    myDobot.goDisSuck(dom.x, dom.y)
    time.sleep(2)
    myDobot.goTop()
    myDobot.rotateRel(-rotate_in_air)
    time.sleep(2)
            
    input("It is your turn, please play and type ENTER...")
    