# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

# =============================================================================
# DEFINE DOMINOES
# =============================================================================
N,S,E,W = 0, 1, 2, 3

class Domino():
    """representation of a domino"""
    # list containing the dominos on the board
    board = []
    def __init__(self, name):
        """Name char exemple : "11","42", ..."""
        self.name = name
        # self.position_x = position_x
        # self.position_y = position_y

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
        
        # define the parent
        if (parent.name not in Domino.board):
            raise Exception("The parent is not on the board!")
        self.parent = parent
       
        # define the children list
        if (self.dom_type == "double"):        # case 3 links (double)
            self.children = ['empty','empty','empty'] # N,S,E the W is the father
        elif (self.dom_type == "simple"):      # case 1 links (simple)
            self.children = 'empty' # the child is in the N

        # define on the dominos which side is north and which is south (see convention)
        if (self.dom_type == "double"): #take care about first put the non 0 side (#give more possibilities)
            self.north = int(name[0])
            self.south = int(name[1])
        else:
            if type(parent) == Starting_Domino: 
                if position == S:
                    if int(name[0]) == self.parent.south:
                        self.south = int(name[0])
                        self.north = int(name[1])
                    elif int(name[1]) == self.parent.south:
                        self.south = int(name[1])
                        self.north = int(name[0])
                    elif int(name[0]) == 0:
                        self.south = int(name[0])
                        self.north = int(name[1])
                    elif int(name[1]) == 0:
                        self.south = int(name[1])
                        self.north = int(name[0])
                    else:
                        raise Exception("The dominos are not compatibles!")
                else:
                    if int(name[0]) == self.parent.north:
                        self.south = int(name[0])
                        self.north = int(name[1])
                    elif int(name[1]) == self.parent.north:
                        self.south = int(name[1])
                        self.north = int(name[0])
                    elif int(name[0]) == 0:
                        self.south = int(name[0])
                        self.north = int(name[1])
                    elif int(name[1]) == 0:
                        self.south = int(name[1])
                        self.north = int(name[0])
                    else:
                        raise Exception("The dominos are not compatibles!")
            else:   # if parent == following domino
                if int(name[0]) == self.parent.north:
                    self.south = int(name[0])
                    self.north = int(name[1])
                elif int(name[1]) == self.parent.north:
                    self.south = int(name[1])
                    self.north = int(name[0])
                elif int(name[0]) == 0:
                    self.south = int(name[0])
                    self.north = int(name[1])
                elif int(name[1]) == 0:
                    self.south = int(name[1])
                    self.north = int(name[0])
                else:
                    raise Exception("The dominos are not compatibles!")
                    
        #Add the child in the parent list
        self.addChild_to_parent(position)

    def addChild_to_parent(self, position):
        """Add the child to the good place on the parent list"""
        if self.parent.dom_type == "simple":
            self.parent.children = self.name
        else:
            self.parent.children[position] = self.name
           
        
        
class Starting_Domino(Domino):
    def __init__(self, name):
        # create the domino
        Domino.__init__(self, name)
        self.north = int(name[0])
        self.south = int(name[1])
    
        #Add the domino to the list of dominoes on the board
        if self.name in Domino.board:
            raise Exception("This domino was already put put on the board!")
        else:
            Domino.board.append(self.name)

        # Define its possible links
        if (self.dom_type == "double"):        # case 4 links (double)
            self.children = ['empty','empty','empty','empty'] # N,S,E,W
        else:                                   # case 2 links (simple)
            self.children = ['empty','empty'] # N,S


# =============================================================================
# Launch a game
# =============================================================================

def draw(hand, stock):
    """the player with this hand draw one tills"""
    if len(stock)==0:
        print("the stock is empty !")
    else : 
        hand = list(hand)
        hand.append(stock[0])
        stock = stock[1:]
    return(np.array(hand), stock)

        
def play_this_domino(name, parent):
    """play the domino in a logical order parent=Domino_obj"""

    if parent.dom_type == "simple":
        if type(parent) == Starting_Domino:
            if parent.north == int(name[0]) or parent.north == int(name[1]) or parent.north == 0:
                if parent.children[N] == 'empty':
                    return(Domino_on_board(name, parent, N))
                else:
                    return("The domino is already fully connected!"+parent.children)
            elif parent.south == int(name[0]) or parent.south == int(name[1]) or parent.south == 0:
                if parent.children[S] == 'empty':
                    return(Domino_on_board(name, parent, S))
                else:
                    return("The domino is already fully connected!"+parent.children)
        else:   #one possibility if following domino
            if parent.children == 'empty':
                return(Domino_on_board(name, parent, N))
            else:
                 return("The domino is already fully connected!"+parent.children)
    
    elif parent.dom_type == "double":
        if type(parent) == Starting_Domino:   #order : N then S then E then W
            if parent.children[N] == 'empty':
                return(Domino_on_board(name, parent, N))
            elif parent.children[S] == 'empty':
                return(Domino_on_board(name, parent, S))
            elif parent.children[E] == 'empty':
               return(Domino_on_board(name, parent, E))
            elif parent.children[W] == 'empty':
                return(Domino_on_board(name, parent, W))
            
            else:
                return("The domino is already fully connected!"+parent.children)

        #order : E then N then S
        else:
            if parent.children[E] == 'empty':
                return(Domino_on_board(name, parent, E))
            elif parent.children[N] == 'empty':
                return(Domino_on_board(name, parent, N))
            elif parent.children[S] == 'empty':
                return(Domino_on_board(name, parent, S))
            else:
                return("The domino is already connected!")

# dom23 = Starting_Domino("23")
# dom24 = play_this_domino("24", dom23)
# dom54 = play_this_domino("54", dom24)

def show_possibilities(hand, board):
    """ Return the play possibles corresponding to a hand :
        the name of the tilts in the hand, the futur parent, the corresponding number
        format : list(name, parent, number)"""
    
    # Get the parent that still have children added
    parent_free_on_board = []
    for domino in board:
        for i in range(0,len(domino.children)):
            if domino.children[i] == "empty" :
                if i == 1:  #for the case parent==starting_domino
                    num = domino.south
                else:
                    num = domino.north
                if [domino, num] not in parent_free_on_board:
                    parent_free_on_board.append([domino, num])
    
    # Get the dominoes that can be added to the parent
    possibles = []
    for domino in hand:
        for domino_b in parent_free_on_board: #parent+number_possible
            f1 = int(domino[0])
            f2 = int(domino[1])
            if f1 == domino_b[1] or f2 == domino_b[1]:
                possibles.append([domino]+domino_b)
            if f1 == 0 or f2 == 0 or domino_b[1] == 0:
                if [domino, num] not in parent_free_on_board:
                    ans = [domino]+domino_b
                    ans[2]=0
                    possibles.append(ans)
                    
    return parent_free_on_board, possibles

def remove_from(array_dom, string):
    n=int(array_dom.shape[0])
    i=0
    while (i < n) and (array_dom[i] != string):
        i = i+1
    new = list(array_dom[0:i])+list(array_dom[i+1:n])
    return(np.array(new))

###Strategies applied by ai
def choose_play_random(possibles):
    """chose randomly among the possible plays"""
    if len(possibles)==0:
        return(False)
    else:
        play = random.choice(possibles)
        #play = possibles[0]
        return(play)
    
#######################################"

"""Start a dominoes game with m tilts par hand"""
m=5

# Start the game
dominoes = np.array(  [ "66",
                        "65","55",
                        "64","54","44",
                        "63","53","43","33",
                        "62","52","42","32","22",
                        "61","51","41","31","21","11",
                        "60","50","40","30","20","10","00" ] )

np.random.shuffle(dominoes)
# dispense tilts 
hand1 = dominoes[0:m]
hand2 = dominoes[m:2*m]
stock = dominoes[2*m:]

#Place the first domino on the board from the stock
Board = []
Board.append(Starting_Domino(stock[0]))
stock = stock[1:]

for i in range(0,1):
    
    if i%2 == 0:
        current_hand = hand1
    else:
        current_hand = hand2
    
    
    print("\n# Values of the game :")
    print(hand1)
    print(hand2)
    print(Board)
    print("\n### Test ###")
    
    possibilities = show_possibilities(current_hand, Board)
    print("# parents_free :")
    print(possibilities[0])
    print("# possibilities :")
    print(possibilities[1])
    
    print("# play chosen : (domino, parent)")
    play = choose_play_random(possibilities[1])
    print(play)
    
    if play == False:
        print("No play available... you draw LOL")
        if i%2 == 0:
            hand1, stock = draw(hand1, stock)
        else:
            hand2, stock = draw(hand2, stock)
    else :
        dom = play_this_domino(play[0], play[1])
        Board.append(dom)
        print("The player play "+str(dom)+" on this domino of the board "+str(play[1]))
        if i%2 == 0:
            hand1, stock = draw(hand1, stock)
            remove_from(hand1, dom.name)
        else:
            hand2, stock = draw(hand2, stock)
            remove_from(hand1, dom.name)
    
    input("Press Enter to continue...")



