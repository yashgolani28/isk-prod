# ----- Imports -------------------------------------------------------

# Standard Imports
import struct
import serial
import time
import numpy as np
import math
import datetime

# Local Imports
from parseFrame import *

#Initialize this Class to create a UART Parser. Initialization takes one argument:
# 1: String Lab_Type - These can be:
#   a. 3D People Counting
#   b. SDK Out of Box Demo
#   c. Long Range People Detection
#   d. Indoor False Detection Mitigation
#   e. (Legacy): Overhead People Counting
#   f. (Legacy) 2D People Counting
# Default is (f). Once initialize, call connectComPorts(self, cliComPort, DataComPort) to connect to device com ports.
# Then call readAndParseUart() to read one frame of data from the device. The gui this is packaged with calls this every frame period.
# readAndParseUart() will return all radar detection and tracking information.
class uartParser():
    def __init__(self,type='SDK Out of Box Demo'):
        self.replay = 0

        if (type == DEMO_NAME_OOB):
            self.parserType = "Standard"
        elif (type == DEMO_NAME_LRPD):
            self.parserType = "Standard"
        elif (type == DEMO_NAME_3DPC):
            self.parserType = "Standard"
        elif (type == DEMO_NAME_SOD):
            self.parserType = "Standard"
        elif (type == DEMO_NAME_VITALS):
            self.parserType = "Standard"
        elif (type == DEMO_NAME_MT):
            self.parserType = "Standard"
        # TODO Implement these
        elif (type == "Replay"):
            self.replay = 1
        else: 
            print ("ERROR, unsupported demo type selected!")
        
        # Data storage
        self.now_time = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    

    def WriteFile(self, data):
        filepath=self.now_time + '.bin'
        objStruct = '6144B'
        objSize = struct.calcsize(objStruct)
        binfile = open(filepath, 'ab+') #open binary file for append
        binfile.write(bytes(data))
        binfile.close()

    # This function is always called - first read the UART, then call a function to parse the specific demo output
    # This will return 1 frame of data. This must be called for each frame of data that is expected. It will return a dict containing all output info
    # Point Cloud and Target structure are liable to change based on the lab. Output is always cartesian.
    def readAndParseUart(self):
        magicWord = bytearray(b'\x02\x01\x04\x03\x06\x05\x08\x07')
        self.fail = 0
        if (self.replay):
            return self.replayHist()
    
        # Find magic word, and therefore the start of the frame
        index = 0
        frameData = bytearray()
        while True:
            magicByte = self.dataCom.read(1)
            if not magicByte:
                continue  # skip empty reads
            if index < len(magicWord) and magicByte[0] == magicWord[index]:
                frameData.append(magicByte[0])
                index += 1
                if index == len(magicWord):
                    break
            else:
                index = 0
                frameData = bytearray()
        
        # Read in version from the header
        versionBytes = self.dataCom.read(4)
        
        frameData += bytearray(versionBytes)

        # Read in length from header
        lengthBytes = self.dataCom.read(4)
        frameData += bytearray(lengthBytes)
        frameLength = int.from_bytes(lengthBytes, byteorder='little')
        
        # Subtract bytes that have already been read, IE magic word, version, and length
        # This ensures that we only read the part of the frame in that we are lacking
        frameLength -= 16 

        # Read the rest of the frame (serial.read may return fewer bytes)
        remaining = frameLength
        while remaining > 0:
            chunk = self.dataCom.read(remaining)
            if not chunk:
                # Incomplete frame: drop and resync on next magic
                return {}
            frameData += bytearray(chunk)
            remaining -= len(chunk)
 
        # frameData now contains an entire frame, send it to parser
        try:
            if self.parserType == "Standard":
                outputDict = parseStandardFrame(frameData)
                try:
                    tt = outputDict.get('tlv_types')
                    if tt:
                        # replace with your logger if available
                        logger(f"[RADAR][TLVs] {tt}")
                    if outputDict.get("ui_mode") == "tm":
                        logger(f"[RADAR][TM] lanes={outputDict.get('tm_viz',{}).get('stats',{}).get('lanes')}")
                except Exception:
                    pass
                if 'trackData' in outputDict and any(obj.get("speed_kmh", 0.0) > 3.0 for obj in outputDict['trackData']):
                    stats = outputDict.get('stats')
                    if stats:
                        print("[Stats TLV]", stats)
            else:
                print('FAILURE: Bad parserType')
                outputDict = {"error": 1}
        except Exception as e:
            print(f"[ParserError] {e}")
            outputDict = {"error": 2}
        
        return outputDict

    # Find various utility functions here for connecting to COM Ports, send data, etc...
    # Connect to com ports
    # Call this function to connect to the comport. This takes arguments self (intrinsic), cliCom, and dataCom. No return, but sets internal variables in the parser object.
    def connectComPorts(self, cliCom, dataCom):
        self.cliCom = serial.Serial(cliCom, 115200,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE,timeout=0.3)
        self.dataCom = serial.Serial(dataCom, 921600,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE,timeout=0.3)
        self.dataCom.reset_output_buffer()
        print('Connected')

    #send cfg over uart
    def sendCfg(self, cfg):
        for line in cfg:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith('%'):
                continue  # skip comments or empty lines

            print(f"[CFG SEND] → {clean_line}")
            self.cliCom.write((clean_line + '\n').encode())

            ack1 = self.cliCom.readline().decode(errors="ignore").strip()
            ack2 = self.cliCom.readline().decode(errors="ignore").strip()
            if ack1:
                print(f"[RADAR RESP] {ack1}")
            if ack2:
                print(f"[RADAR RESP] {ack2}")

            time.sleep(0.05)

        time.sleep(2)
        self.cliCom.reset_input_buffer()
        self.cliCom.close()
        print("[INFO] Config sent successfully.")

    #send single command to device over UART Com.
    def sendLine(self, line):
        self.cliCom.write(line.encode())
        ack = self.cliCom.readline()
        print(ack)
        ack = self.cliCom.readline()
        print(ack)

    # def replayHist(self):
    #     if (self.replayData):
    #         #print('reading data')
    #         #print('fail: ',self.fail)
    #         #print(len(self.replayData))
    #         #print(self.replayData[0:8])
    #         self.replayData = self.Capon3DHeader(self.replayData)
    #         #print('fail: ',self.fail)
    #         return self.pcBufPing, self.targetBufPing, self.indexes, self.numDetectedObj, self.numDetectedTarget, self.frameNum, self.fail, self.classifierOutput
    #         #frameData = self.replayData[0]
    #         #self.replayData = self.replayData[1:]
    #         #return frameData['PointCloud'], frameData['Targets'], frameData['Indexes'], frameData['Number Points'], frameData['NumberTracks'],frameData['frame'],0, frameData['ClassifierOutput'], frameData['Uniqueness']
    #     else:
    #         filename = 'overheadDebug/binData/pHistBytes_'+str(self.saveNum)+'.bin'
    #         #filename = 'Replay1Person10mShort/pHistRT'+str(self.saveNum)+'.pkl'
    #         self.saveNum+=1
    #         try:
    #             dfile = open(filename, 'rb', 0)
    #         except:
    #             print('cant open ', filename)
    #             return -1
    #         self.replayData = bytes(list(dfile.read()))
    #         if (self.replayData):
    #             print('entering replay')
    #             return self.replayHist()
    #         else:
    #             return -1
        
def getBit(byte, bitNum):
    mask = 1 << bitNum
    if (byte&mask):
        return 1
    else:
        return 0