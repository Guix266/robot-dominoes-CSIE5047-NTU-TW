from serial.tools import list_ports
from pydobot import Dobot
from pydobot.message import Message
import struct
import threading
import time
import warnings
import serial

class DobotDominoes():
    def __init__(self):
        self.port = list_ports.comports()[0].device
        self.device = Dobot(port=self.port, verbose=True)
        self.device.speed(100,100)

    def updatePose(self):
        (self.x,self.y,self.z,self.r,self.j1,self.j2,self.j3,self.j4)=self.device.pose()
        
    def setHome(self,x,y,z,r):
        self.msg = Message()
        self.msg.id = 30
        self.msg.ctrl = 0x03
        self.msg.params = bytearray([])
        self.msg.params.extend(bytearray(struct.pack('f', x)))
        self.msg.params.extend(bytearray(struct.pack('f', y)))
        self.msg.params.extend(bytearray(struct.pack('f', z)))
        self.msg.params.extend(bytearray(struct.pack('f', r)))
        self.device._send_message(self.msg)
        
    def goHome(self):
        self.msg = Message()
        self.msg.id = 31
        self.msg.ctrl = 0x03
        self.device._send_message(self.msg)
        time.sleep(30)
        self.updatePose()
    
    def printPose(self):
        self.updatePose()
        print((self.x,self.y,self.z,self.r,self.j1,self.j2,self.j3,self.j4))
    
    def moveAbs(self,x,y,z):
        #self.device.move_to(x, y, z, self.r, wait=True) #MODE_PTP_MOVL_XYZ = 0x02
        self.device._set_ptp_cmd(x,y,z,self.r,mode=0x01,wait=True) #MODE_PTP_MOVJ_XYZ = 0x01
        #self.updatePose()
    
    def moveRel(self,x,y,z):
        self.updatePose()
        self.device.move_to(self.x+x, self.y+y, self.z+z, self.r, wait=True)
        #self.updatePose()
    
    def rotateAbs(self,r):
        self.updatePose()
        self.device.move_to(self.x, self.y, self.z, r, wait=True)
        #self.updatePose()
        
    def rotateRel(self,r):
        self.updatePose()
        self.device.move_to(self.x, self.y, self.z, self.r + r, wait=True)
        #self.updatePose()
        
    def enSuck(self):
        self.device.suck(True)
    
    def disSuck(self):
        self.device.suck(False)
    
    def goTop(self):
        self.moveAbs(200,0,170)

    def goTopHand(self):
        #self.moveAbs(200,-100,170)
        #self.moveAbs(150,-150,170)
        #self.moveAbs(100,-200,170)
        self.moveAbs(0,-200,170)
        
    #def goBack(self):
        #self.moveAbs(100,-200,170)
        #self.moveAbs(150,-150,170)
        #self.moveAbs(200,-100,170)
        #self.moveAbs(200,0,170)

    def goSuck(self,x,y):
        self.updatePose()
        self.moveAbs(x,y,22)
        self.enSuck()
        self.moveAbs(x,y,6)
        self.moveAbs(x,y,22)
    
    def goDisSuck(self,x,y):
        self.updatePose()
        self.moveAbs(x,y,22)
        self.disSuck()
  
# myDobot=DobotDominoes()      
# # _set_ptp_cmd(x, y, z, r, mode=MODE_PTP_MOVL_XYZ, wait=wait)
# myDobot.goHome()
# myDobot.goTopHand()
#device.move_to(165, 0, 195, 0, wait=True)
#device.move_to(20, y, z, r, wait=True)
#device.move_to(x, y, z, r, wait=True)  # we wait until this movement is done before continuing