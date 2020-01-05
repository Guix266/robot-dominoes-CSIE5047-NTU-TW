# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 10:49:14 2020

@author: guix
"""

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
        if (self.dom_type == "double"):  #take care about connect first the non 0 side 
            self.north = int(name[0])               # (#give more possibilities)
            self.south = int(name[1])
        else: #domino simple
            if type(parent) == Starting_Domino: 
                if position == S:
                    # Classical case
                    if int(name[0]) == self.parent.south:
                        self.south = int(name[0])
                        self.north = int(name[1])
                    elif int(name[1]) == self.parent.south:
                        self.south = int(name[1])
                        self.north = int(name[0])
                    # case where the new domino is connected to the 0 of the old one 
                    # Give arbitrary the south and north
                    elif self.parent.south==0:
                        self.south = int(name[0])
                        self.north = int(name[1])
                    # case where new domino has 0
                    elif int(name[0]) == 0:         
                        self.south = int(name[0])
                        self.north = int(name[1])
                    elif int(name[1]) == 0:
                        self.south = int(name[1])
                        self.north = int(name[0])
                
                    else:
                        raise Exception("The dominos are not compatibles!")
                else:
                    # Classical case
                    if int(name[0]) == self.parent.north:
                        self.south = int(name[0])
                        self.north = int(name[1])
                    elif int(name[1]) == self.parent.north:
                        self.south = int(name[1])
                        self.north = int(name[0])
                    # case where the new domino is connected to the 0 of the old one 
                    # Give arbitrary the south and north
                    elif self.parent.south==0:
                        self.south = int(name[0])
                        self.north = int(name[1])
                    # case where new domino has 0
                    elif int(name[0]) == 0:
                        self.south = int(name[0])
                        self.north = int(name[1])
                    elif int(name[1]) == 0:
                        self.south = int(name[1])
                        self.north = int(name[0])
                    else:
                        raise Exception("The dominos are not compatibles!")
            else:   # if parent == following domino
                # Classical case
                if int(name[0]) == self.parent.north:
                    self.south = int(name[0])
                    self.north = int(name[1])
                elif int(name[1]) == self.parent.north:
                    self.south = int(name[1])
                    self.north = int(name[0])
                # case where the new domino is connected to the 0 of the old one 
                # Give arbitrary the south and north
                elif self.parent.north==0:
                    self.south = int(name[0])
                    self.north = int(name[1])
                # case where new domino has 0
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
        
        # angle :
        if self.dom_type == "double":
            self.angle = self.parent.angle + 90
        else:
            if position == S:
                if self.south > self.north:
                    if self.parent.south <= self.parent.north:
                        self.angle = self.parent.angle
                    else:
                        self.angle = self.parent.angle + 180
                else:
                    if self.parent.south <= self.parent.north:
                        self.angle = self.parent.angle + 180
                    else:
                         self.angle = self.parent.angle
            else:
                 if self.south > self.north:
                     if self.parent.south < self.parent.north:
                         self.angle = self.parent.angle + 180
                     else:
                         self.angle = self.parent.angle
                 else:
                     if self.parent.south <= self.parent.north:
                         self.angle = self.parent.angle
                     else:
                         self.angle = self.parent.angle + 180
        
        print("possition :"+ str(position))
        ag = 0
        length = 0
        big_length = 53
        little_length = 40
        # Coordinates
        if self.dom_type == "simple":
            if self.parent.south > self.parent.north:
                if position == N:
                    ag = self.parent.angle + 180
                    length = 2*big_length
                elif position == S:
                    ag = self.parent.angle
                    length = 2*big_length
                elif position == E:
                    ag = self.parent.angle + 90
                    length = big_length + little_length
                elif position == W:
                    ag = self.angle -90
                    length = big_length + little_length
            elif self.parent.south <= self.parent.north:
                if position == N:
                    ag = self.parent.angle
                    length = 2*big_length
                elif position == S:
                    ag = self.parent.angle + 180
                    length = 2*big_length
                elif position == E:
                    ag = self.angle -90
                    length = big_length + little_length
                elif position == W:
                    ag = self.parent.angle + 90
                    length = big_length + little_length
        elif self.dom_type == "double":
            if self.parent.south >= self.parent.north:
                if position == N:
                    ag = self.parent.angle + 180
                    length = big_length + little_length
                elif position == S:
                    ag = self.parent.angle
                    length = big_length + little_length
            elif self.parent.south < self.parent.north:
                if position == N:
                    ag = self.parent.angle
                    length = big_length + little_length
                elif position == S:
                    ag = self.parent.angle + 180
                    length = big_length + little_length
            
        print("ag ="+ str(ag))
        self.x = self.parent.x + (length)*np.cos(ag *np.pi/180)
        self.y = self.parent.y + (length)*np.sin(ag *np.pi/180)

    def addChild_to_parent(self, position):
        """Add the child to the good place on the parent list"""
        if self.parent.dom_type == "simple" and type(self.parent) == Domino_on_board :
            self.parent.children = self.name
        else:
            self.parent.children[position] = self.name
           


class Starting_Domino(Domino):
    def __init__(self, name, start_X, start_Y, start_angle):
        # create the domino
        Domino.__init__(self, name)
        self.north = int(name[0])
        self.south = int(name[1])
    
        # Coordinates
        self.x = start_X
        self.y = start_Y
        self.angle = start_angle
        
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

def play_this_domino(name, parent):
    """play the domino in a logical order parent=Domino_obj
    Only works when the point of connextion is not 0"""

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
                    return("The domino is already fully connected!"+ str(parent.children))
            # case where we connect a domino parerent!=0 children==0
            # We arbitrary connect to north or south of the parent
            else:   
                if parent.children[N] == 'empty':
                    return(Domino_on_board(name, parent, N))
                if parent.children[S] == 'empty':
                    return(Domino_on_board(name, parent, S))
                else:
                    return("The domino is already fully connected!"+ str(parent.children))
                
        else:   #one possibility if following domino
            if parent.children == 'empty':
                return(Domino_on_board(name, parent, N))
            else:
                 return("The domino is already fully connected!"+ str(parent.children))
    
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
                return("The domino is already fully connected!"+ str(parent.children))

        #order : E then N then S
        else:
            if parent.children[E] == 'empty':
                return(Domino_on_board(name, parent, E))
            elif parent.children[N] == 'empty':
                return(Domino_on_board(name, parent, N))
            elif parent.children[S] == 'empty':
                return(Domino_on_board(name, parent, S))
            else:
                return("The domino is already fully connected!"+ str(parent.children))

# dom23 = Starting_Domino("23")
# dom24 = play_this_domino("24", dom23)
# dom54 = play_this_domino("54", dom24)

def show_possibilities(hand, board):
    """ Return the play possibles corresponding to a hand :
        the name of the tilts in the hand, the futur parent, the corresponding number
        format : list(name, parent, number, side of connection along the parent)"""
    
    # Get the parent that still have children added
    parent_free_on_board = []
    for domino in board:
        # case where only one children
        if type(domino.children) == str: 
            if domino.children == "empty" :
                num =  domino.north
                if [domino, num] not in parent_free_on_board:
                        parent_free_on_board.append([domino, num])
        # case where several children
        else:
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
            ans = [domino]+domino_b
            if f1 == domino_b[1] or f2 == domino_b[1]:
                if ans not in possibles:      
                    possibles.append(ans)
            if f1 == 0 or f2 == 0 or domino_b[1] == 0:
                ans[2]=0
                if ans not in possibles:      
                    possibles.append(ans)
                    
    return parent_free_on_board, possibles

###Strategies applied by ai
def choose_play_random(possibles):
    """chose randomly among the possible plays"""
    if len(possibles)==0:
        return(False)
    else:
        play = random.choice(possibles)
        #play = possibles[0]
        return(play)

def biggest_cost_domino(possibles):
    """take in arg a list of possibilities and return the biggest cost domino""" 
    cost = int(possibles[0][0][0]) + int(possibles[0][0][1])
    domino = possibles[0]
    for dom in possibles[1:]:
        c = int(dom[0][0]) + int(dom[0][1])
        if c > cost:
            cost = c
            domino = dom
    return(domino)

def doubles(possibles):
    """take in arg a list of possibilities and return the list of the doubles dominoes""" 
    double =[]
    for dom in possibles:
        if int(dom[0][0]) == int(dom[0][1]):
            double.append(dom)
    return(double)

def better_play(possibles):
    """chose a good play:
    Priority : 1 - play first the non joker tilts (keep them)
               2 - Play doubles first (allows less flexibility)
               3 - play high numbers first (get rid of malus)
        """
    if len(possibles)==0:
        return(False)
    else:
        ### Separate the dominoes that don't connect with 0 first
        joker = []
        no_joker = []
        for poss in possibles:
            if int(poss[0][0])!=0 and int(poss[0][1])!=0:
                no_joker.append(poss)
            else:
                joker.append(poss)
        # first we work on no_joker
        if no_joker != []:
            double = doubles(no_joker)
            if double != []:
                return(biggest_cost_domino(double))
            else:
                return(biggest_cost_domino(no_joker))
        # and then on joker
        else:
            return(biggest_cost_domino(joker))


#######################################"

def draw(hand, stock):
    """the player with this hand draw one tills"""
    if len(stock)==0:
        print("the stock is empty !")
    else : 
        hand = list(hand)
        hand.append(stock[0])
        stock = stock[1:]
    return(np.array(hand), stock)

def remove_from(array_dom, string):
    n=int(array_dom.shape[0])
    i=0
    while (i < n) and (array_dom[i] != string):
        i = i+1
    new = list(array_dom[0:i])+list(array_dom[i+1:n])
    return(np.array(new))

def print_game_situations(hand1, hand2, Board):
    print("\n # Game :")
    print("hand 1 : "+ str(hand1))
    print("hand 2 : "+ str(hand2))
    print("Dominoes on Board : "+ str(Board))

def print_results(hand1, hand2):
    score1 = 0
    score2 = 0
    for domino in hand1:
        score1 += int(domino[0])
        score1 += int(domino[1])
    for domino in hand2:
        score2 += int(domino[0])
        score2 += int(domino[1])    
    print("Player 1 gets "+str(score1)+" malus" )
    print("Player 2 gets "+str(score2)+" malus" )
    return(score1, score2)
    

"""Start a dominoes game with m tilts par hand"""
m=3

# Start the game
dominoes = np.array(  [ "22",
                        "21",
                        "20","10","00" ] )

np.random.shuffle(dominoes)
# dispense tilts 
hand1 = dominoes[0:m]
hand2 = dominoes[m:2*m]
stock = dominoes[2*m:]

#Place the first domino on the board from the stock
Board = []
Board.append(Starting_Domino("11", 150, 150, 0))
# stock = stock[1:]

# i = 0
# while hand1.shape[0] > 0 and hand2.shape[0] > 0:
for i in range(0,3):   
    print_game_situations(hand1, hand2, Board)
    
    if i%2 == 0:
        player = 1
        current_hand = hand1
    else:
        player = 2
        current_hand = hand2
    
    #######################" DESCISIONS
    parent_free_on_board, possibles = show_possibilities(current_hand, Board)
    # print("\n# parents_free :")
    # for elem in parent_free_on_board:
    #     print(elem)
    # print("# possibilities :")
    # for elem in possibles:
        # print(elem)

    ## Choose among the possibilities
    play = better_play(possibles)
    
    print(play)
    print("\n#########################################")
    if play == False:
        print("No play available for the player "+str(player))
        if stock.shape[0] > 0 :
            if i%2 == 0:
                print("The player draws")
                hand1, stock = draw(hand1, stock)
            else:
                print("The player draws")
                hand2, stock = draw(hand2, stock)
    else :
        print("The player " + str(player) + " plays [ "+str(play[0][0])+" | "+str(play[0][1])+" ] on "+str(play[1]))
        
        dom = play_this_domino("00", play[1])
        Board.append(dom)
        
        # Refresh the hands
        if i%2 == 0:
            hand1 = remove_from(hand1, dom.name)
            if stock.shape[0] > 0 :
                hand1, stock = draw(hand1, stock)
        else:
            hand2 = remove_from(hand2, dom.name)
            if stock.shape[0] > 0 :
                hand2, stock = draw(hand2, stock)
    print("#########################################")

    i += 1
    # input("[INFO] Press for next turn...")

print("\n #### Game finished ####")
print_results(hand1, hand2)


