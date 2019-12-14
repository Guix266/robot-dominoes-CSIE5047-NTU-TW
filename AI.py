# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class Domino:

    dominoes = []

    def __init__(self, parent, lower_half):
        self.parent = parent
        self.parent.addChild(self)
        self.children = []
        self.lower_half = lower_half
        self.upper_half = self.parent.lower_half
        self.name = ''.join(sorted([str(self.lower_half), str(self.upper_half)]))
        if self.name in Domino.dominoes:
            raise Exception("This domino was already initialized!")
        else:
            Domino.dominoes.append(self.name)
        if self.lower_half == self.upper_half:
            self.max_children = 3
        else:
            self.max_children = 1

    def __repr__(self):
        return "[ %s | %s ]" % (self.upper_half, self.lower_half)

    def addChild(self, child):
        if len(self.children) >= self.max_children:
            raise Exception("Can't add another child!")
        self.children.append(child)


class StartingDomino(Domino):
    def __init__(self, num):
        self.children = []
        self.lower_half = num
        self.upper_half = num
        self.name = str(num) + str(num)
        self.max_children = 4
        if self.name in Domino.dominoes:
            raise Exception("This domino was already initialized!")
        else:
            Domino.dominoes.append(self.name)


dom11 = StartingDomino(num=1)
dom12 = Domino(parent=dom11, lower_half=2)
dom13 = Domino(parent=dom11, lower_half=3)