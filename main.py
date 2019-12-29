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

# import the modules that we have created
import image_recognition
import AI_v2
# import arm

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
myDobot=arm.DobotDominoes()
myDobot.arm.setHome(200,0,170,0) # x, y, z, theta


# =============================================================================
# # II) Recognition of the first Board
# =============================================================================

#*** Camera take a picture

#for the test import an image
folder  = "images"
filename = "uni_test.png"
img = cv.imread(os.path.join(folder,filename),1)
plt.imshow(img)

# Recognition
result = image_recognition.find_dominoes(img, 1)

#*** function to get real coordinates of the dom
# start_X, start_Y, start_angle = 

# Add the domino to the list of dominoes on the board
domino = result[0]
dom = AI_v2.Starting_Domino(domino[0], start_X, start_Y, start_angle)
Board.append(dom)


# =============================================================================
# # III) Robot go on top of his hand
# =============================================================================

#***


# =============================================================================
# # IV) Robot choose the adapted tilt
# =============================================================================

#*** Camera take a picture

result = image_recognition.find_dominoes(img, 5)

robot_hand = []
for domino in result:
    robot_hand.append(domino[0])

# Compute all the possible plays of robot
parent_free_on_board, possibles = AI_v2.show_possibilities(robot_hand, Board)
#Choose the most adapted play
play = AI_v2.better_play(possibles) # play = ["11", dom_parent, num_connection]


# Add to the list this dominoes
if play == False:
    print("No play available for the robot. ")
    print("The robot need to draw")
else :
    print("The robot plays [ "+str(play[0][0])+" | "+str(play[0][1])+" ] on "+str(play[1]))
    
    dom = AI_v2.play_this_domino(play[0], play[1])
    Board.append(dom)
    
# Get the possition of the domino in the hand
for domino in result:
    if domino[0] == play[0]:
        play_hand = domino      # play_hand = ["11", (X_hand, Y_hand, angle_hand), phi]

# =============================================================================
# # V) Robot need to pick the tilt
# =============================================================================

#*** convert the coordinates in """"play_hand"""" to robot coordinates
    
#*** pick the tilt to the good coordinates

# =============================================================================
# # VI) Robot go back in front of the board and place the tilt
# =============================================================================

#*** pick the tilt to the good coordinates """"contained in dom.x, dom.y, dom.angle""""
# ATTENTION NEED TO RESCALE THE ANGLE BETWEEN -180,180
# I could modify it in AI_v2 if needed


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


