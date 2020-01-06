from arm import *

# initialization
myDobot=DobotDominoes()
myDobot.setHome(200,0,170,0) # x, y, z, theta
myDobot.goHome()
myDobot.printPose()

myDobot.goTop()
# move to a position to suck the domino, and then release it
myDobot.goSuck(200,0) # only consider (x, y) ignore (z, theta)
myDobot.goDisSuck(200,0)

myDobot.rotateAbs(-90) # theta
 
# for taking a picture
myDobot.goTop()
myDobot.moveAbs(0,-200,170)

myDobot.goTopHand()

myDobot.goSuck(200,0)
myDobot.rotateAbs(0)

myDobot.goDisSuck(-85.82128643357206,-304.8988626105795)
