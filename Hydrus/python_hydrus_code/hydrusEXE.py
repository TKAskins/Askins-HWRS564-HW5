# HYDRUS_Class.py
# Created by Derek Groenendyk
# created 10/16/2011
# modified 12/6/2019
# Class that runs HYDRUS, modifies the input files, and arranges the output data.

import os
from scipy.io import savemat
import shutil
import subprocess
import sys
import time

from hydrus.outfiles import NODINFFAST

# sys.stdout = open('run_hydrus.out', 'w')

class HYDRUS:
    
    def __init__(self,install_dir,ExpFileLocation,exp):
        self.directory = ExpFileLocation
        self.exp = exp
        # list of file extensions of interest
        self.fileExtList = [".out",".in",".txt",".dat",".pkl"]

        self.install_dir = install_dir

        try:
            os.makedirs(str(self.directory))
        except OSError:
            pass            

        sys.stdout = open(str(self.directory / 'run_hydrus.out'), 'w')

    #runs the HYDRUS program    
    def run_hydrus(self,noCMDWindow):

        args = str(self.install_dir / 'H1D_CALC.exe')

        args2 = [args,str(self.directory)]

        out = subprocess.call(args2, stderr=sys.stdout, stdout=sys.stdout)

    # modifies the HYDRUS LEVEL_01.DIR file
    # not needed if you provide the experiment location to HYDRUS
    def update_LEVEL01(self,newExpFileLocation = None):
        if newExpFileLocation == None:
            newExpFileLocation = str(self.directory)
        try:
            infile = open(str(self.install_dir / "LEVEL_01.DIR"),"r+")
        except IOError:
            root = Tk()
            root.withdraw()
            createLEVEL01 = tkMessageBox.askyesno("Warning!","LEVEL_01.DIR file does not exist!! Would you like to create it now?")
            if createLEVEL01:
                infile = open(str(self.install_dir / "LEVEL_01.DIR"),"w")
            else:
                exit()
        
        infile.write(newExpFileLocation)
        infile.close()

    #moves the output files and creates output folders
    def outputResults(self,resultsDir,trial=None):

        if trial != None:
            resultsDir = str(resultsDir / trial)
        
        INFILES = []
        
        os.chdir(self.directory)
        self.currentDir = os.curdir
        self.dirList = os.listdir(self.currentDir)
        
        #adds all the files of interest to INFILES
        for OUTFILE in self.dirList:
            if os.path.splitext(OUTFILE)[1].lower() in self.fileExtList:
                INFILES.append(OUTFILE)
                      
        #informs user if folders already exist, and asks if they should be overwritten.
        try:
            os.makedirs(resultsDir)
        except OSError:
            while True:
                try:
                    shutil.rmtree(resultsDir)
                except WindowsError:
                    time.sleep(10)
                else:
                    os.makedirs(resultsDir)
                    break

        #moves all the files to a directory
        for infile in INFILES:
##            print infile
            shutil.copy2(self.directory+ "\\" + infile, resultsDir)
##            shutil.move(self.directory+ "\\" + infile,resultsDir)

        return resultsDir

    def saveOutput(self,resultsDir,trial=None, add_time_flag=True, get_hdata=False, get_fdata=True):

        #Initialize Classes
        time0 = time.time()

        nodInf = NODINFFAST(self.directory)
        time1 = time.time()

        wcdata, hdata, fdata = nodInf.readData()
        times = nodInf.getTimes()

        time2 = time.time()

        dataDict = {'wc':wcdata}
        if add_time_flag:
            dataDict['t'] = times
        if get_hdata:
            dataDict['p'] = hdata
        if get_fdata:
            dataDict['f'] = fdata

        #informs user if folders already exist, and asks if they should be overwritten.
        try:
            os.makedirs(str(resultsDir))
        except OSError:     
            print('directory already exists')       

        savemat(str(resultsDir / ('Trial'+str(trial)+'.mat')), mdict={'data':dataDict}, do_compression=True)

        time3 = time.time()
