# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:52:01 2019

@author: guix
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# import the modules that we have created
import image_recognition
import AI_v2
import arm

# =============================================================================
# 1rst round
# =============================================================================

# The robot go on top of the board and recognise the first domino
#***

# Recognition of the board
#*** Camera take a picture
result = image_recognition.find_dominoes(img, 1)


#Place the first domino on the board from the stock
Board = []
Board.append(AI_v2.Starting_Domino(stock[0], 290, 0, 90))
stock = stock[1:]

image_recognition.find_dominoes(img, N)