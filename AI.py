# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# class Domino:

#     dominoes = []

#     def __init__(self, parent, lower_half):
#         self.parent = parent
#         self.parent.addChild(self)
#         self.children = []
#         self.lower_half = lower_half
#         self.upper_half = self.parent.lower_half
#         self.name = ''.join(sorted([str(self.lower_half), str(self.upper_half)]))
#         if self.name in Domino.dominoes:
#             raise Exception("This domino was already initialized!")
#         else:
#             Domino.dominoes.append(self.name)
#         if self.lower_half == self.upper_half:
#             self.max_children = 3
#         else:
#             self.max_children = 1

#     def __repr__(self):
#         return "[ %s | %s ]" % (self.upper_half, self.lower_half)

#     def addChild(self, child):
#         if len(self.children) >= self.max_children:
#             raise Exception("Can't add another child!")
#         self.children.append(child)


# class StartingDomino(Domino):
#     def __init__(self, num):
#         self.children = []
#         self.lower_half = num
#         self.upper_half = num
#         self.name = str(num) + str(num)
#         self.max_children = 4
#         if self.name in Domino.dominoes:
#             raise Exception("This domino was already initialized!")
#         else:
#             Domino.dominoes.append(self.name)


# dom11 = StartingDomino(num=1)
# dom12 = Domino(parent=dom11, lower_half=2)
# dom13 = Domino(parent=dom11, lower_half=3)



import numpy as np
import matplotlib.pyplot as plt
import random as r

# =============================================================================
# definition variables / simulation of the game
# =============================================================================

# tilt number in each hand
m = 5

dominos = np.array([[6,6],
                   [6,5],[5,5],
                   [6,4],[5,4],[4,4],
                   [6,3],[5,3],[4,3],[3,3],
                   [6,2],[5,2],[4,2],[3,2],[2,2],
                   [6,1],[5,1],[4,1],[3,1],[2,1],[1,1],
                   [6,0],[5,0],[4,0],[3,0],[2,0],[1,0],[0,0]] )

# shuffle dominos
np.random.shuffle(dominos)

# dispense tilts 
hand1 = dominos[0:m]
hand2 = dominos[m:2*m]
stock = dominos[2*m:]

class Dominos:
    """class which define a domino"""
    def __init__(self, parent, name, openhalf):
        self.child = []
        self.parent = parent
        self.name = name
        openhalf = openhalf
        self.possible_connexion = []
    

domino = Dominos("12", "23", 2)

# =============================================================================
# Game operation 
# =============================================================================

# definition of the board game
board = stock[0]
stock = stock[1:]

def turn(hand, board):
    """play what the player with this hand has to do"""





