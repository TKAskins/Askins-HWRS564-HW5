#IN_Class.py
#Created by Derek Groenendyk
#10/17/2011
#This a class that accesses the files with the Extension .IN

import os
# from Tkinter import *
# import tkMessageBox
from hydrus.readline import *
import numpy as np

class SELECTORIN:
    
    def __init__(self,directory):
        self.directory = directory
        os.chdir(str(self.directory))
        self.dirList = os.listdir(str(self.directory))
        # File of interest
        self.IN_index = self.dirList.index("SELECTOR.IN")
        
        self.readLines()
        
    # update init information    
    def update(self):
        self.dirList = os.listdir(str(self.directory))
        os.chdir(str(self.directory))
        self.IN_index = self.dirList.index("SELECTOR.IN")

        outfile = open(self.dirList[self.IN_index],"w")
        outfile.seek(0,0)                    
        outfile.writelines(self.lines)
        outfile.close()
        
        self.readLines()

    # read file information and save info to init
    def readLines(self):
        #reads in the lines of the file     
        infile = open(str(self.directory / self.dirList[self.IN_index]),"r")
        self.lines = infile.readlines()
        infile.close()

        # time, length, and mass units
        units = ["LUnit","TUnit","MUnit"]
        self.data = []

        for i in [5,6,7]:
            self.data.append([units[i-5],self.lines[i].split()[0]])

        numMaterials = int(self.lines[13].split()[0])

        offset = 0
        for i in [8,10,12,15,17,19,21,23,25,28,30,32]:  # need to adapt for mutiple materials
            # print i
##            if i > 25+numMaterials:
##                offset = numMaterials - 1
            headings = self.lines[i+offset].split()
            values = self.lines[i+1+offset].split()
            if i == 15:
                for j in range(len(headings)-6):
                    self.data.append([headings[j],values[j]])
            elif i == 25:
                for j in range(len(headings)):
                    values = []
                    offset = 1
                    while self.lines[i+offset].split()[0] != '***': # multiple materials, needs fixin' -9/2/2014 DGG
                        val = self.lines[i+offset].split()[j]
                        values.append(val)
                        offset += 1
                    info = [headings[j]]
                    for k in range(len(values)):
                        info.append(values[k])
                    self.data.append(info)
            else:
                for j in range(len(headings)):
                    self.data.append([headings[j],values[j]])
            

        printTimes = []
        for i in range(41,len(self.lines)-1):
            line = self.lines[i].split()
            if line[0] == '***':
                break
            for time in line:
                printTimes.append(time)

    def addData(self,aType=0,cType=0,t=0,lEnsemble='f'):
        
##        outfile = open(self.dirList[self.IN_index],"r+")

        paramValues = [t,lEnsemble,cType,aType]
        paramList = ['Trial','Ensemble','CropType','iAssim']
        paramDict = dict(zip(paramList,paramValues))
        
        for param in paramDict.keys():
            if param in self.lines[12].split():
                self.setData(param,paramDict[param])
            else:
                self.lines[12] = self.lines[12][:22]+'  iAssim'+3*' '+'CropType'+3*' '+'Ensemble'+3*' '+'Trial'+'\n'
                self.lines[13] = self.lines[13][:19]+8*' '+str(aType)+9*' '+str(cType)+9*' '+lEnsemble+10*' '+str(t)+'\n'
                break

##        wline = appendToLine(self.lines[i],paramIndex,paramValue)
##        self.lines[i+1] = wline 

##        outfile.seek(0,0)                    
##        outfile.writelines(self.lines)
##        outfile.close()
                
    def addRootBlock(self, P0=-10, POT=-25, P2H=-200, P2L=-800,
                     P3=-8000, r2H=0.5, r2L=0.1, POptm=-25):
        for line in self.lines:
            if 'ROOT WATER' in line or 'iRootIn' in line:
                return  # block already present

        for i in range(len(self.lines) - 1, -1, -1):
            if '*** END' in self.lines[i]:
                end_idx = i
                break

        block_f = [
            '*** BLOCK F: ROOT WATER UPTAKE INFORMATION  ****************************\n',
            'iRootIn    iMoistInF  iRFak   iNoObs\n',
            '      0          0      0       0\n',
            'P0      POT     P2H     P2L     P3     r2H     r2L    POptm(1)\n',
            f'  {P0}    {POT}    {P2H}    {P2L}   {P3}    {r2H}     {r2L}      {POptm}\n',
        ]

        for j, line in enumerate(block_f):
            self.lines.insert(end_idx + j, line)

        self.update()

    def addMat(self,numMat=1):
        headings = self.lines[25].split()
        values = self.lines[25+1].split()
        print(len(self.lines))
        [self.lines.insert(25+1,self.lines[25+1]) for i in range(numMat-1)]
        print(len(self.lines))

        self.update()
                        

    # return init data            
    def getData(self,param,pInd=1):
        for data in self.data:
            if data[0] == param:
                # print(data)
                paramValue = data[1]
                if param in ['thr','ths','Alfa','n','Ks']:
                    # print(data)
                    paramValue = [data[pInd]]

        return paramValue[0]

    # set and write in new data
    def setData(self,param,paramValue,mat = 1):
        
##        self.update()

        if param == 'TPrint(1),TPrint(2),...,TPrint(MPL)':
            for i in range(len(self.lines)):
                if 'BLOCK D:' in self.lines[i]:
                    endLine = i
                    break
                if '*** END' in self.lines[i]:
                    endLine = i
                    break
            for i in range(len(self.lines)):
                if param in self.lines[i].split():
                    wline = ''
                    line = 0
                    for j in range(len(paramValue)):
                        if j % 6 != 0:
                            wline += ' '*(12-len(str(paramValue[j])))+str(paramValue[j])
                        else:
                            if line+i < endLine and line > 0:
                                self.lines[i+line] = wline + '\n'
                            elif line > 0:
                                self.lines.insert(i+line,wline+'\n')
                            wline = ' '*(11-len(str(paramValue[j])))+str(paramValue[j])
                            line += 1
##                    if j-1 % 6 != 0: line += 1
                    if line+i < endLine and line > 0:
                        self.lines[i+line] = wline + '\n'
                    elif line > 0:
                        self.lines.insert(i+line,wline+'\n')
                    if i+line < endLine:
                        [self.lines.pop(i) for i in np.arange(endLine-1,i+line,-1)]
                    break
                
        elif param in ['thr','ths','Alfa','n','Ks','l']:
            for i in range(len(paramValue)):
                self.writeGenData(param,paramValue[i],inc=mat-1)
                
        elif param in ['Trial','Ensemble','CropType','iAssim']:
            paramDict = {'iAssim':0,'CropType':0,'Trial':0,'Ensemble':'f'}
            paramDict[param] = paramValue
            
            if param in self.lines[12].split():
                self.writeGenData(param,paramValue)
            else:
                self.lines[12] = self.lines[12][:22]+'  iAssim'+3*' '+'CropType'+3*' '+'Ensemble'+3*' '+'Trial'+'\n'
                self.lines[13] = self.lines[13][:19]+8*' '+str(paramDict['iAssim'])+9*' '+str(paramDict['CropType'])+\
                9*' '+paramDict['Ensemble']+10*' '+str(paramDict['Trial'])+'\n'

        elif param in ['xRMax']:
            paramDict = {'xRMax':0}
            paramDict[param] = paramValue
            if param in self.lines[78].split():
                self.writeGenData(param,paramValue)
            else:
                print('Change line reference for xRMax')

        else:
            i = -1
            for data in self.data:
                i += 1
                if data[0] == param:
                    self.data[i][1] = paramValue
            self.writeGenData(param,paramValue)          

    # write data back to file
    def writeGenData(self,param,paramValue,inc=0):

##        outfile = open(self.dirList[self.IN_index],"r+")
        
        for i in range(len(self.lines)):
            if param in self.lines[i].split():
                paramIndex = self.lines[i].split().index(param)
##                print 'inc: ',inc
                wline = writeLine(self.lines[i+1+inc],paramIndex,paramValue)
                self.lines[i+1+inc] = wline 

##        outfile.seek(0,0)                    
##        outfile.writelines(self.lines)
##        outfile.close()

    def writeData(self,param,paramValue):
        
        outfile = open(self.dirList[self.IN_index],"r+") 

        for i in range(len(self.lines)):
            if param in self.lines[i].split():
                
                line = self.lines[i+1].split()
                for j in range(len(line)):
                    if j == 0:
                        whitespace = 7 - len(str(line[j]))
                        line[j] = " "*whitespace + str(line[j])
                    elif j == 4:
                        whitespace = 9 - len(str(int(paramValue)))
                        line[j] = " "*whitespace + str(paramValue)
                    else:
                        whitespace = 8 - len(str(line[j]))
                        line[j] = " "*whitespace + str(line[j])
                wline = "%s%s%s%s%s%s"% \
                (line[0],line[1],line[2],line[3],line[4],line[5])
                self.lines[i+1] = wline + "\n"

        outfile.seek(0,0)                    
        outfile.writelines(self.lines)
        outfile.close()



class ATMOSPHIN:
    
    def __init__(self,directory):
        
        self.directory = directory
        os.chdir(str(self.directory))
        self.dirList = os.listdir(str(self.directory))        
        # File of interest
        self.IN_index = self.dirList.index("ATMOSPH.IN")
        
        self.readLines()
       
    def update(self):
        self.dirList = os.listdir(str(self.directory))   
        os.chdir(str(self.directory))
        self.IN_index = self.dirList.index("ATMOSPH.IN")

        outfile = open(self.dirList[self.IN_index],"w")
        outfile.seek(0,0)                    
        outfile.writelines(self.lines)
        outfile.close()
        
        self.readLines()

    def readLines(self):
        #reads in the lines of the ATMOSPH.IN file  
        infile = open(str(self.directory / self.dirList[self.IN_index]),"r")
        self.lines = infile.readlines()
        infile.close()

        self.data = []

        for i in [2,4,6]:
            headings = self.lines[i].split()
            values = self.lines[i+1].split()
            if i == 4:
                for j in range(2):
                    self.data.append([headings[j],values[j]])
            else:
                for j in range(1):
                    self.data.append([headings[j],values[j]])
                    
        headings = self.lines[8].split()
        for j in range(len(headings)):
            self.data.append([headings[j],'0.0'])

        read_flag = False
        self.numLines = 0
        time_data = []
        for i,line in enumerate(self.lines):
            self.numLines += 1
            if line.split()[0].lower() == 'tatm':
                read_flag = True
                continue            
            if line.split()[0].lower() == 'end***':
                read_flag = False
                self.numLines -= 1
                break
            if read_flag:                
                values = [float(item) for item in line.split()]                
                time_data.append(values)

        # numTimes = len(self.lines) - 11 
        # values = np.zeros((numTimes,len(self.lines[8].split())-1))
        # for i in range(numTimes):
        #     lineVals = self.lines[i+9].split()
        #     values[i,:] = [float(item) for item in lineVals]
            
        self.atmoData = values

    def getNumTimes(self):
        # return int(self.lines[-2].split()[0])
        return int(self.numLines)

    def getData(self,param):
        for data in self.data:
            if data[0] == param:
                paramValue = data[1]

        return paramValue


    def setData(self,param,paramValue,time=None):
        
##        self.update()

##        i = -1
##        for data in self.data:
##            i += 1
##            if data[0] == param:
##                if i > 3:
##                    for j in range(4,len(self.data)):
##                        if self.data[j][0] == param:
##                            self.data[j][1] = paramValue
##                        else:
##                            self.data[j][1] = ['0']*len(paramValue)
##                else:
##                    self.data[i][1] = paramValue
##                    break

        self.writeGenData(param,paramValue,time)

    def delTimes(self,num_del_lines):
        print("deleting lines...")
        self.update()
        outfile = open(self.dirList[self.IN_index],"w")
        # numLines = self.getNumTimes() - end + 1 

        lastLine = self.lines[-1]
        self.lines = self.lines[:-(num_del_lines + 1)]
        self.lines.append(lastLine)
        
        outfile.seek(0,0)                    
        outfile.writelines(self.lines)
        outfile.close()
        self.update()

    def insertData(self,start,times):
        outfile = open(self.dirList[self.IN_index],"w")
        numLines = len(self.lines)
        for i in range(len(times)):
##            print self.lines[start+8] 
            wline = writeLine(self.lines[start+8],0,str(times[i]))
            self.lines.insert(start+8, wline)

        outfile.seek(0,0)                    
        outfile.writelines(self.lines)
        outfile.close()

        self.update()        
        

    def addTimes(self,times):
##        self.update()
        outfile = open(self.dirList[self.IN_index],"w")

        start = self.numLines - len(self.atmoData)
        end = start + len(times)
        
        # numLines = len(self.lines)
        # start = int(self.lines[-2].split()[0])
        print(start,end)
        for i,t in enumerate(times):
            wline = writeLine(self.lines[-2],0,t)
            self.lines.insert(len(self.lines)-1, wline)

        outfile.seek(0,0)                    
        outfile.writelines(self.lines)
        outfile.close()

        self.update()


    def writeGenData(self,param,paramValue,time):
    # maybe make an option to keep old data?
    # just add new line? insert new line? remove line?

##        outfile = open(self.dirList[self.IN_index],"w")
        write_flag = False
        for i in range(len(self.lines)):
            sline = self.lines[i].split()
##            print 'atmosph.in line: '+str(i)
            if sline[0] == 'end***':
                break
            if param in sline:
                paramIndex = sline.index(param)
                write_flag = True
                print(paramIndex)
                continue

            if write_flag:
                if time != None:
                    print(float(sline[0]) == time)
                    if time == float(sline[0]):
                        self.lines[i] = writeLine(self.lines[i],paramIndex,paramValue)
                else:
                    self.lines[i] = writeLine(self.lines[i],paramIndex,paramValue)
##                elif i == 8:
##                    self.atmoData[time,paramIndex] = paramValue
                    
        self.update()

##        outfile.seek(0,0)                    
##        outfile.writelines(self.lines)
##        outfile.close()



class METEOIN:
    
    def __init__(self,directory):
        
        self.directory = directory
        os.chdir(str(self.directory))
        self.dirList = os.listdir(str(self.directory))
        # File of interest
        self.IN_index = self.dirList.index("METEO.IN")
        
        self.readLines()
       
    def update(self):
        os.chdir(str(self.directory))
        self.dirList = os.listdir(str(self.directory))
        self.IN_index = self.dirList.index("METEO.IN")

        outfile = open(self.dirList[self.IN_index],"w")
        outfile.seek(0,0)                    
        outfile.writelines(self.lines)
        outfile.close()
        
        self.readLines()

    def readLines(self):
        #reads in the lines of the METEO.IN file
        infile = open(str(self.directory / self.dirList[self.IN_index]),"r")        
        self.lines = infile.readlines()
        infile.close()

        self.data = []

        self.iRad = int(self.lines[3].split()[1])
        icrop = None
        if self.iRad == 0:
            for i in [2,4,6,8,10,12,14,16,18,20,21,23]:
                headings = self.lines[i].split()
                values = self.lines[i+1].split()
                if i == 4:
                    for j in range(2):
                        self.data.append([headings[j],values[j]])
                elif i == 16:
                    headings = [headings[0],headings[10],headings[11]]
                    icrop = values[0]
                    for j in range(3):
                        self.data.append([headings[j],values[j]])
                elif icrop == '3' and i > 16:
                    if i == 18:
                        headings = [headings[0],headings[10]]
                        for j in range(2):
                            self.data.append([headings[j],values[j]])
                    elif i == 20:
                        for j in range(len(headings)):
                            self.data.append([headings[j],values[j]])
##                    if i == 22:
##                        headings = [headings[0]]
##                        for j in range(1):
##                            self.data.append([headings[j],values[j]])
                    elif i == 23:
                        for j in range(len(headings)):
                            self.data.append([headings[j],'0.0'])
                elif icrop == '0' and i > 16:
                    if i == 18:
                        for j in range(len(headings)):
                            self.data.append([headings[j],values[j]])
                    elif i == 21:
                        for j in range(len(headings)):
                            self.data.append([headings[j],'0.0'])
                else:
                    for j in range(len(headings)):
                        self.data.append([headings[j],values[j]])
                        
            if len(self.lines) > 30:
                self.numTimes = int(self.lines[-3].split()[0])
                if icrop == '3':
                    offset = 25
                if icrop == '0':
                    offset = 23
                
                values = np.zeros((self.numTimes,len(self.lines[offset].split())))
                for i in range(self.numTimes):
                    lineVals = self.lines[i+offset].split()
                    values[i,:] = [float(item) for item in lineVals]
                self.atmoData = values
            else:
                self.numTimes = 0
            
        elif self.iRad == 2:
            for i in [2,4,6,8,10,12,13,15]:
                headings = self.lines[i].split()
##                if i != 17:
                values = self.lines[i+1].split()
                if i == 4:
                    for j in range(2):
                        self.data.append([headings[j],values[j]])
                elif i == 8:
                    headings = [headings[0],headings[10],headings[11]]
                    icrop = values[0]
                    for j in range(3):
                        self.data.append([headings[j],values[j]])
                elif icrop == '3' and i > 8:
                    if i == 10:
                        headings = [headings[0],headings[10]]
                        for j in range(2):
                            self.data.append([headings[j],values[j]])
                    elif i == 12:
                        for j in range(len(headings)):
                            self.data.append([headings[j],values[j]])
                    elif i == 15:
                        for j in range(len(headings)):
                            self.data.append([headings[j],'0.0'])
                elif icrop == '0' and i > 8:
                    if i == 10:
                        for j in range(len(headings)):
                            self.data.append([headings[j],values[j]])
                    elif i == 13:
                        for j in range(len(headings)):
                            self.data.append([headings[j],'0.0'])
                else:
                    for j in range(len(headings)):
                        self.data.append([headings[j],values[j]])
                        
            if len(self.lines) > 18:
                self.numTimes = int(self.lines[-3].split()[0])
                if icrop == '3':
                    offset = 17
                if icrop == '0':
                    offset = 15
                    
                values = np.zeros((self.numTimes,len(self.lines[offset].split())))
                for i in range(self.numTimes):
                    lineVals = self.lines[i+offset].split()
                    # not sure why the if/else, DGG 7/30/2013
                    if len(lineVals) > len(self.lines[offset-2].split()):
                        values[i,:] = [float(lineVals[j]) for j in range(len(self.lines[offset-2].split()))]
                    else:
                        values[i,:] = [float(item) for item in lineVals]
                self.atmoData = values
            else:
                 self.numTimes = 0

    def getNumTimes(self):
        return self.numTimes

    def getData(self,param):
        for data in self.data:
            if data[0] == param:
                paramValue = data[1]

        return paramValue

    def addData(self):
        outfile = open(self.dirList[self.IN_index],"w")

        if self.iRad == 0: index = 23
        elif self.iRad == 2:
            if self.getData('iCrop') == '3':
                index = 15
            elif self.getData('iCrop') == '0':
                index = 13        

        if 'DOY' in self.lines[index].split():
            numTimes = int(self.lines[-2].split()[0])
            if len(self.lines[index+2].split()) < 12:
                for i in range(numTimes):
                    self.lines[i+index+2] = self.lines[i+index+2][:-1] + '   0' +'\n'
        else:
            self.lines[index] = self.lines[index][:-1] + 3*' '+ 'DOY\n'
            self.lines[index+1] = self.lines[index+1][:-1] + 4*' '+ '[d]\n'
            numTimes = len(self.lines) - index+4
                
            for i in range(numTimes):
                self.lines[i+index+2] = self.lines[i+index+2][:-1] + '   0' +'\n'
                
##        wline = appendToLine(self.lines[i],paramIndex,paramValue)
##        self.lines[i+1] = wline 

        outfile.seek(0,0)                    
        outfile.writelines(self.lines)
        outfile.close()
        self.update()

    def addLines(self,end):
        self.update()
        outfile = open(self.dirList[self.IN_index],"w")
        if self.getNumTimes() != 0:
            start = int(self.lines[-2].split()[0])
        else:
            start = 0  
        t = start
        for i in range(end - start):
            t += 1
            if self.getNumTimes() != 0:
                wline = writeLine(self.lines[-2],0,t)
            else:
                blankLine = '       0         0           0           0         0         0        0          0             0            0         0 '
                wline = writeLine(blankLine,0,t)
            self.lines.insert(len(self.lines)-1, wline)
    
        outfile.seek(0,0)                    
        outfile.writelines(self.lines)
        outfile.close()

##        outfile = open(self.dirList[self.IN_index][:-8]+'temp.txt',"w")
##        outfile.writelines(self.lines)
##        outfile.close()

        self.update()

    def delInterception(self,irad,albedo,icrop):
        self.update()
        if irad == 2:
            ind = 10
        elif irad == 0:
            ind = 18
        [self.lines.pop(i) for i in [ind,ind,ind,ind]]
        self.lines.insert(ind,' Albedo\n')
        self.lines.insert(ind+1,'   '+str(albedo)+'\n')
        self.lines[ind-1] = '          '+str(icrop)+' '*48+'0         1\n'
        outfile = open(self.dirList[self.IN_index],"w")
        outfile.seek(0,0)                    
        outfile.writelines(self.lines)
        outfile.close()
        self.update()

    def delLines(self,end):
        print("deleting lines...")
        self.update()
        outfile = open(self.dirList[self.IN_index],"w")
        numLines = self.getNumTimes() - end + 1 
        
        lastLine = self.lines[-1]
        self.lines = self.lines[:-numLines]
        self.lines.append(lastLine)

        print(numLines,self.getNumTimes())
        print(lastLine)
        
        outfile.seek(0,0)                    
        outfile.writelines(self.lines)
        outfile.close()
        self.update()        

    def setData(self,param,paramValue,time=None,fname=None):
##        self.update()

        if param == 'Radiation':
            self.iRad = paramValue
##            if self.iRad == 0:  index = 23
##            elif self.iRad == 2:  index = 15
                
            infile = open(fname,'r')
            data = infile.readlines()
            infile.close()
            data[-1] += '\n'
            
##            tempIRad = int(self.lines[3].split()[1])
##            if tempIRad == 0:  index = 23
##            elif tempIRad == 2:  index = 15
            for i in range(len(self.lines)):
                if self.lines[i].split()[0] == '[T]':
                    index = i               
            
            timeData = self.lines[index+1:]
            self.lines = data+timeData
            
            self.update()
            
        self.writeGenData(param,paramValue,time)

    def writeGenData(self,param,paramValue,time):
    # maybe make an option to keep old data?
    # just add new line? insert new line? remove line?

##        outfile = open(self.dirList[self.IN_index],"w")

        for i in range(len(self.lines)):
            if param in self.lines[i].split():
                paramIndex = self.lines[i].split().index(param)
                if time != None:
                    self.lines[i+time+2] = writeLine(self.lines[i+time+2],paramIndex,paramValue)
                else:
                    self.lines[i+1] = writeLine(self.lines[i+1],paramIndex,paramValue)
##                elif i == 8:
##                    self.atmoData[time,paramIndex] = paramValue
                    
                    

##        outfile.seek(0,0)                    
##        outfile.writelines(self.lines)
##        outfile.close()
        self.update()



class PROFILEDAT:
    
    def __init__(self,directory):
        
        self.directory = directory
        os.chdir(str(self.directory))
        self.dirList = os.listdir(str(self.directory))
        # File of interest
        self.IN_index = self.dirList.index("PROFILE.DAT")
        
        self.readLines()
       
    def update(self):
        os.chdir(str(self.directory))
        self.dirList = os.listdir(str(self.directory))
        self.IN_index = self.dirList.index("PROFILE.DAT")

        outfile = open(self.dirList[self.IN_index],"w")
        outfile.seek(0,0)                    
        outfile.writelines(self.lines)
        outfile.close()
        
        self.readLines()

    def readLines(self):
        #reads in the lines of the "PROFILE.DAT" file
        infile = open(str(self.directory / self.dirList[self.IN_index]),"r")
        self.lines = infile.readlines()
        infile.close()

        self.data = []

        self.numlayers = int(self.lines[4].split()[0])


    def getData(self,param):
        for data in self.data:
            if data[0] == param:
                paramValue = data[1]

        return paramValue 

    def setData(self,param,paramValue,layer=None):            
        self.writeGenData(param,paramValue,layer)

    def writeGenData(self,param,paramValue,layer):
        if param.lower() in ['mat', 'lay']:
            paramValue = str(int(paramValue))
        else:
            paramValue = '{0:0.6e}'.format(paramValue)        

        for i in range(len(self.lines)):
            if param in self.lines[i].split():
                paramIndex = self.lines[i].split().index(param)-3
                if layer != None:
                    self.lines[i+layer+1] = writeLine(self.lines[i+layer+1],paramIndex,paramValue)
                else: # does all lines
                    while self.lines[i+1].split()[0] != '0':
                        self.lines[i+1] = writeLine(self.lines[i+1],paramIndex,paramValue)
                        i += 1

        self.update()

    def setBeta(self, root_depth_cm):
        for i in range(len(self.lines)):
            if 'Beta' in self.lines[i].split():
                header_idx = i
                beta_col = self.lines[i].split().index('Beta') - 3
                break

        node = 1
        while self.lines[header_idx + node].split()[0] != '0':
            depth_cm = abs(float(self.lines[header_idx + node].split()[1]))
            beta_val = '{0:0.6e}'.format(1.0 if depth_cm <= root_depth_cm else 0.0)
            self.lines[header_idx + node] = writeLine(
                self.lines[header_idx + node], beta_col, beta_val)
            node += 1

        self.update()


    def addLayers(self,layers,head=None):

        # if self.lines[-1].split()[0] != str(self.numlayers):
        #     lastLine = len(self.lines) - 2
        # else:
        #     lastLine = len(self.lines) - 1

        for i,depth in enumerate(layers):
            if type(head) == list or type(head) == np.ndarray:
                lyr_head = head[i]
            else:
                lyr_head = head

            templine = self.lines[self.numlayers + 4]
            templine = writeLine(templine,0,int(self.numlayers) + 1)
            if depth > 0.:
                depth *= -1.
            templine = writeLine(templine,1,'{0:0.6e}'.format(depth))
            templine = writeLine(templine,2,'{0:0.6e}'.format(lyr_head))
            self.lines.insert(self.numlayers + 5, templine)
            self.numlayers += 1

        bottom = layers[-1]
        if bottom > 0.:
            bottom *= -1

        self.lines[3] = writeLine(self.lines[3],1,'{0:0.6e}'.format(bottom))
        self.lines[4] = writeLine(self.lines[4],0,self.numlayers)

        self.update()



    def delLayers(self,num_del_layers):

        lastLines = self.lines[-1:]

        self.lines = self.lines[:-(num_del_layers + len(lastLines))]
        bottom = self.lines[-1].split()[1]
        self.lines.extend(lastLines)

        self.numlayers -= num_del_layers

        self.lines[3] = writeLine(self.lines[3],1,bottom)
        self.lines[4] = writeLine(self.lines[4],0,self.numlayers)
        

        self.update()

        #         [paramValue,exps] = profile.readExp(paramValue);
        #         if ~isempty(ind)
        #             kk = 0;
        #             for jj=6:5+length(paramValue)
        #                 kk = kk + 1;
        #                 sLine = textscan(profile.C{jj}, '%s','whitespace','');
                        
        #                 if exps(jj-5) >= 0.
        #                     val = sprintf('%fe+%03d',paramValue(jj-5),exps(jj-5));
        #                 elseif exps(jj-5) < 0.
        #                     val = sprintf('%fe%04d',paramValue(jj-5),exps(jj-5));
        #                 end
                        
        #                 wline = writeLine(sLine{1},ind,val);
        #                 profile.C{jj} = wline;
        #             end
        #         end
        #     end
        # end
        
        # function [A,B] = readExp(profile,data)
        #     sData = sprintf('%e,',data);
        #     str1 = sprintf('%%%d.%df',8,6);
        #     results = textscan(sData,strcat(str1,' %2s  %03d'),'delimiter',',');
        #     A = results{1};
        #     B = results{3};
        # end
            


class DATAIN:
    
    def __init__(self,directory):
        self.directory = directory
        os.chdir(str(self.directory))
        self.dirList = os.listdir(str(self.directory))
        # File of interest
        self.IN_index = self.dirList.index("DATA.IN")

        self.readLines()

    def update(self):
        os.chdir(str(self.directory))
        self.dirList = os.listdir(str(self.directory))
        self.IN_index = self.dirList.index("DATA.IN")

        outfile = open(self.dirList[self.IN_index],"w")
        outfile.seek(0,0)                    
        outfile.writelines(self.lines)
        outfile.close()
        
        self.readLines()


    def readLines(self):
        #reads in the lines of the "DATA.IN" file
        infile = open(str(self.directory / self.dirList[self.IN_index]),"r")        
        self.lines = infile.readlines()
        infile.close()

        for i in range(len(self.lines)):
            if len(self.lines[i]) > 0:
                if '*CROPPARAMETERS' in self.lines[i].split():
                    headings1 = self.lines[i+1].split()
                    data1 = self.lines[i+2].split()
                if '*ASSIMILATION' in self.lines[i].split():
                    headings2 = self.lines[i+1].split()
                    data2 = self.lines[i+2].split()
                    for i in range(len(headings2)):
                        headings1.append(headings2[i])
                        data1.append(data2[i])
                    break

                                  
        self.data = zip(headings1,data1)
        

    def getData(self,param):
        for data in self.data:
            if data[0] == param:
                paramValue = data[1]

        return paramValue 


    def setData(self,param,paramValue,heading='*CROPPARAMETERS'):
        self.writeGenData(param,paramValue,heading)

    def writeGenData(self,param,paramValue,heading):
        for i in range(len(self.lines)):
            if len(self.lines[i]) > 0:
                if heading in self.lines[i].split():
                    paramIndex = self.lines[i+1].split().index(param)
                    self.lines[i+2] = writeLine(self.lines[i+2],paramIndex,paramValue)

        self.update()

class ASSIMIN:
    
    def __init__(self,directory):
        self.directory = directory
        os.chdir(str(self.directory))
        self.dirList = os.listdir(str(self.directory))
        # File of interest
        self.IN_index = self.dirList.index("ASSIM.IN")

        self.readLines()

    def update(self):
        os.chdir(str(self.directory))
        self.dirList = os.listdir(str(self.directory))
        self.IN_index = self.dirList.index("ASSIM.IN")

        outfile = open(self.dirList[self.IN_index],"w")
        outfile.seek(0,0)                    
        outfile.writelines(self.lines)
        outfile.close()
        
        self.readLines()


    def readLines(self):
        #reads in the lines of the "ASSIM.IN" file
        infile = open(str(self.directory / self.dirList[self.IN_index]),"r")        
        self.lines = infile.readlines()
        infile.close()

        for i in range(len(self.lines)):
            if len(self.lines[i]) > 0:
                if '*ASSIMILATION' in self.lines[i].split():
                    headings = self.lines[i+1].split()
                    data = self.lines[i+2].split()
                    break
                                  
        self.data = zip(headings,data)
        
    def getData(self,param):
        for data in self.data:
            if data[0] == param:
                paramValue = data[1]

        return paramValue 


    def setData(self,param,paramValue,heading='*ASSIMILATION'):            
        self.writeGenData(param,paramValue,heading)

    def writeGenData(self,param,paramValue,heading):
        for i in range(len(self.lines)):
            if len(self.lines[i]) > 0:
                if heading in self.lines[i].split():
                    paramIndex = self.lines[i+1].split().index(param)
                    self.lines[i+2] = writeLine(self.lines[i+2],paramIndex,paramValue)

        self.update()


class ASSIMDATA:
    
    def __init__(self,directory,filename='assimdata.txt'):
        self.directory = directory
        os.chdir(str(self.directory))
        self.dirList = os.listdir(str(self.directory))
        infile = open(str(self.directory / self.dirList[self.IN_index]),"r")
##        print filename
        # File of interest
        self.IN_index = self.dirList.index(filename)

        self.readLines()

    def update(self):
        os.chdir(str(self.directory))
        self.dirList = os.listdir(str(self.directory))
        self.IN_index = self.dirList.index("assimdata.txt")

        outfile = open(self.dirList[self.IN_index],"w")
        outfile.seek(0,0)                    
        outfile.writelines(self.lines)
        outfile.close()
        
        self.readLines()


    def readLines(self):
        #reads in the lines of the "assimdata.txt" file
        infile = open(str(self.directory / self.dirList[self.IN_index]),"r")        
        self.lines = infile.readlines()
        infile.close()

        headings = self.lines[0].split()

        data = np.zeros((len(headings),len(self.lines)-1))
        for i in range(1,len(self.lines)):
            if len(self.lines[i]) > 0:
                data[:,i-1] = [float(item) for item in self.lines[i].split()]
                        
        self.data = zip(headings,[data[heading,:] for heading in range(len(headings))])
       

    def getData(self,param):
        for data in self.data:
            if data[0] == str(float(param)):
                paramValue = data[1]

        return paramValue 
