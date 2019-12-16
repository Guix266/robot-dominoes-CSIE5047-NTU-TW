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
import matplotlib as mpl
import matplotlib.pyplot as plt

# =============================================================================
# DEFINE DOMINOES
# =============================================================================
dominos = np.array(  [  "66",
                        "65","55",
                        "64","54","44",
                        "63","53","43","33",
                        "62","52","42","32","22",
                        "61","51","41","31","21","11"])
                        # "60","50","40","30","20","10","00"] )

N,S,E,W = 0, 1, 2, 3

class Domino():
    """representation of a domino"""
    # list containing the dominos on the board
    board = []
    def __init__(self, name):
        """Name char exemple : "11","42""..."""
        self.name = name

        if int(name[0])==int(name[1]):
            self.dom_type = "double"
        else:
            self.dom_type = "simple"
        
    # return a fancy representation of the domino
    def __repr__(self):
        return( "[ %s | %s ]" % (int(self.name[0]),int(self.name[1])))


class Domino_on_board(Domino):
    """class of the dominoes on the board"""
    
    def __init__(self, name, parent, position):
        """The input is the domino we want to put on the board and its father
        and position if several possible (N, S, E)"""
        # create the domino
        Domino.__init__(self, name)
        
        #Add the domino to the list of dominoes on the board
        if self.name in Domino.board:
            raise Exception("This domino was already put put on the board!")
        else:
            Domino.board.append(self.name)
        
        #Test if it is possible to add this domino to this parent
        self.test_compatibility(name, parent, position)
            
        # define the parent
        if (parent.name not in Domino.board):
            raise Exception("The parent is not on the board!")
        self.parent = parent
        raise Exception("The parent is not on the board!")
        # define the children list
        if (self.dom_type == "double"):        # case 3 links (double)
            self.children = ['empty','empty','empty'] # N,S,E the W is the father
            self.value = int(name[0])
        else:                                   # case 1 links (simple)
            self.children = 'empty' # the child is in the N
            self.north = int(name[0])
            self.south = int(name[1])
        
        #Add the child in the parent list
        # self.addChild_to_parent(position)
        self.parent.children[position] = self.name

    # def addChild_to_parent(self, position):
    #     """need childs an position if several possible (N, S, E)"""
    #     if self.parent.dom_type == "simple":
    #         self.parent.children = self.name
    #     else:
    #         self.parent.children[position] = self.name

    def test_compatibility(self, name, parent, position):
        if name[0] != parent.name[0] and name[0] != parent[1] and name[1] != parent[0] and name[1] != parent[1]:
            raise Exception("These 2 dominoes are not compatible, you can't play "+ str(Domino(name))+ " with "+str(Domino(parent)))
                        
        
class Starting_Domino(Domino):
    def __init__(self, name):
        # create the domino
        Domino.__init__(self, name)
    
        #Add the domino to the list of dominoes on the board
        if self.name in Domino.board:
            raise Exception("This domino was already put put on the board!")
        else:
            Domino.board.append(self.name)

        # Define its possible links
        if (self.dom_type == "double"):        # case 4 links (double)
            self.children = ['empty','empty','empty','empty'] # N,S,E,W
            self.value = int(name[0])
        else:                                   # case 2 links (simple)
            self.children = ['empty','empty'] # N,S
            self.north = int(name[0])
            self.south = int(name[1])
        
        

dom11 = Starting_Domino("11")
dom12 = Domino_on_board("13", dom11, N)


# =============================================================================
# Launch a game
# =============================================================================

np.random.shuffle(dominos)
print(dominos)

# tilt number in each hand
m = 4

# dispense tilts 
hand1 = dominos[0:m]
hand2 = dominos[m:2*m]
stock = dominos[2*m:]

def draw(hand, stock):
    """the player with this hand draw one tills"""
    if len(stock)==0:
        print("the stock is empty !")
    else : 
        hand.append(stock[0])
        stock = stock[1:]

#def options(hand)
        
def play_this_domino(name, parent):
    """play the domino in a logical order parent=Domino_obj"""
    
    if parent.dom_type == "simple":
        if parent.type == Starting_Domino:
            if parent.north == int(name[0]) or parent.north == int(name[1]):
                if parent.children[N] == 'empty':
                    Domino_on_board(name, parent, N)
                else:
                    return("The domino is already fully connected!"+parent.children)
            elif parent.south == int(name[0]) or parent.south == int(name[1]):
                if parent.children[S] == 'empty':
                    Domino_on_board(name, parent, N)
                else:
                    return("The domino is already fully connected!"+parent.children)
        else:   #one possibility if following domino
            if parent.children == 'empty':
                Domino_on_board(name, parent, N)
            else:
                 return("The domino is already fully connected!"+parent.children)
    
    elif parent.dom_type == "double":
        if parent.type == Starting_Domino:   #order : N then S then E then W
            if parent.children[N] != 'empty':
                Domino_on_board(name, parent, N)
            elif parent.children[S] != 'empty':
                Domino_on_board(name, parent, S)
            elif parent.children[E] != 'empty':
                Domino_on_board(name, parent, E)
            elif parent.children[W] != 'empty':
                Domino_on_board(name, parent, W)
            else:
                return("The domino is already fully connected!"+parent.children)

        #order : E then N then S
        else:
            if parent.children[E] != 'empty':
                Domino_on_board(name, parent, E)
            elif parent.children[N] != 'empty':
                Domino_on_board(name, parent, N)
            elif parent.children[S] != 'empty':
                Domino_on_board(name, parent, S)
            else:
                return("The domino is already connected!")

    