## HYDRUSmain.py
## Written by Derek Groenendyk
##Updated by Trevor Askins
## original: 03/30/2012
## Final Base: 12/06/2019
## TA updates 06/06/2026
## Calls and modifies CropModel project


#TA: import standard and project-specific libraries used for file operations, numerical computation, and HYDRUS control
import numpy as np
import os
from pathlib import Path
import shutil
import time
from tqdm import tqdm

from hydrusEXE import HYDRUS
from utils.wc import WC
from hydrus.infiles import PROFILEDAT,SELECTORIN,ATMOSPHIN
from hydrus.readline import writeLine


#TA: determine the current working directory used for locating HYDRUS projects and results
current_folder = Path(os.getcwd())

#TA: main function that configures and executes HYDRUS simulation workflows based on the selected method
def run():

    #TA: configure the HYDRUS executable directory, experiment type, and project file location
    hydrus_exe_dir = Path(r'C:\Users\Trevor\OneDrive\Documents\UoA\Hydrogeology\Semester 2\Research\Hydrus')

    #TA: Choose experiment type
    exp = 'Flood'
    # exp = 'Drip'

    #TA: Choose plant type
    # PlantType = 'corn'
    PlantType = 'beans'
    # PlantType = 'none'
    
    #TA: Identify the experiment File location
    ExpFileLocation = current_folder / "HYDRUS_Projects" / exp
    noCMDWindow = 1

    #TA: configure root water uptake and root growth in HYDRUS1D.DAT based on plant type
    root_flag = 0 if PlantType == 'none' else 1
    setHydrus1DParams(ExpFileLocation, {
        'RootWaterUptake': root_flag,
        'RootGrowth': root_flag,
        'InitialCondition': root_flag,
    })

    profile_depth_map = {'corn': '2.E+02', 'beans': '6.E+01', 'none': '2.E+02'}
    setHydrus1DParams(ExpFileLocation, {'ProfileDepth': profile_depth_map[PlantType]})

    rroot_map = {'corn': 1.5, 'beans': 1.0, 'none': 0}
    ATMOSPHIN(ExpFileLocation).setData('rRoot', rroot_map[PlantType], time=None)

    if PlantType in ('corn', 'beans'):
        ht_row1  = 12 if PlantType == 'corn' else 10
        tAtm_row3 = 15 if PlantType == 'corn' else 7
        setAtmosphRows(ExpFileLocation, [
            {'ht': ht_row1},
            {'ht': ht_row1 // 2},
            {'tAtm': tAtm_row3, 'ht': 0},
        ])

    if PlantType == 'corn':
        setSelectorParams(ExpFileLocation, {
            'MPL': 15,
            'tMax': 15,
            'TPrint(1),TPrint(2),...,TPrint(MPL)': list(range(1, 16)),
            'P0': -15, 'P2H': -325, 'P2L': -600, 'P3': -8000,
            'r2H': 0.5, 'r2L': 0.1,
            'POptm(1),POptm(2),...,POptm(NMat)': -30,
        })
    elif PlantType == 'beans':
        setSelectorParams(ExpFileLocation, {
            'MPL': 7,
            'tMax': 7,
            'TPrint(1),TPrint(2),...,TPrint(MPL)': list(range(1, 8)),
            'P0': -15, 'P2H': -750, 'P2L': -2000, 'P3': -8000,
            'r2H': 0.5, 'r2L': 0.1,
            'POptm(1),POptm(2),...,POptm(NMat)': -30,
        })

    #TA: initialize the HYDRUS controller with the executable path and experiment project
    hydrusEXE = HYDRUS(hydrus_exe_dir, ExpFileLocation, exp)

    startTime = time.time()

    results_folder = 'daily'
    results_dir = current_folder / 'results' / exp / results_folder

    profile = PROFILEDAT(ExpFileLocation)
    atmo = ATMOSPHIN(ExpFileLocation)

    # ensure Block G (root water uptake) is present in SELECTOR.IN
    SELECTORIN(ExpFileLocation).addRootBlock()

    soil_offset = 0
    sample_offset = 0
    nsamples = 1
    #TA: instead of searching parameters, just create a list defining the parameters for my soils directly
    paramValues, _ = getAllTexParams(prct='five_twenty', model='new', lmean=True, llog=False, num=nsamples - 1)

    numSoils = paramValues.shape[0]

    paramValues[:, 4] = np.round(paramValues[:, 4] / 24.0, 4) # cm/hr from cm/day

    # texture lookup: same order as five_twenty in wc.py
    texture_list = []
    for sand in range(20, 65, 5):
        for silt in range(20, 81 - sand, 5):
            texture_list.append((sand, silt, 100 - sand - silt))

    if PlantType == 'corn':
        root_depths = range(0, 210, 10)   # 0, 10, 20, ... 200 cm
    elif PlantType == 'beans':
        root_depths = range(0, 65, 5)    # 0, 5, 10, ... 60 cm
    else:
        root_depths = range(0, 210, 10)   # default: 0, 10, 20, ... 200 cm

    for soil in tqdm(range(soil_offset, soil_offset + numSoils)):

        paramDict = {
            'lSink': 't' if root_flag else 'f',
            'lRoot': 't' if root_flag else 'f',
            'hTabN':100000}

        setSelectorParams(ExpFileLocation, paramDict)

        for ind in range(sample_offset, nsamples):

            paramList = ['thr', 'ths', 'Alfa', 'n', 'Ks']
            paramDict = dict(zip(['thr', 'ths', 'Alfa', 'n', 'Ks'], range(5)))

            data = [[str(paramValues[soil, paramDict[paramList[i]], ind])] for i in range(len(paramList))]
            dataDict = dict(zip(paramList, data))

            setSelectorParams(ExpFileLocation, dataDict)

            data = np.squeeze(np.array(data, dtype=float))

            # InitCond=t means h column is water content (theta), not pressure head
            th_mid = (data[0] + data[1]) / 2.0
            profile.setData('h', th_mid)

            for root_depth in root_depths:

                atmo.setData('RootDepth', root_depth)
                hydrusEXE.run_hydrus(noCMDWindow)

                sand, silt, clay = texture_list[soil]
                trial_num = f"{exp}{sand:02d}.{silt:02d}.{clay:02d}.rd{root_depth}.{PlantType}"

                hydrusEXE.saveOutput(results_dir, trial=trial_num, add_time_flag=True, get_hdata=True, get_fdata=True)

        print('Finished iteration: '+str(soil)+'.'+str(ind))

    print('Done...')

    stopTime = time.time()
    lengthOfRun = stopTime - startTime
    timeOfDay = time.localtime(time.time())
    print('Took:   '+str(lengthOfRun)+' seconds,  Finished at:  '+str(timeOfDay[3])+':'+str(timeOfDay[4]))

#TA: helper function to retrieve soil structure or saturation conductivity data from the water content utility
def getAllSSC():

    water = WC(current_folder)
    paramValues = SSCData = water.getSSC() 

    return paramValues

#TA: helper function to retrieve filtered texture parameters and index values used by complex simulation workflows
def getAllTexParams(prct='two', model='old', num=0, lmean=True, llog=False, amy_flag=False):

    water = WC(current_folder)
    paramValues, idx = water.get_params(prct, model, lmean, llog=llog, num=num, lnotnan=True, amy=amy_flag)

    if len(paramValues.shape) < 3:
        paramValues = paramValues[:, :, np.newaxis]

    return paramValues, idx    

#TA: helper function to update key=value parameters in HYDRUS1D.DAT
def setHydrus1DParams(ExpFileLocation, paramDict):
    dat_path = ExpFileLocation / 'HYDRUS1D.DAT'
    with open(str(dat_path), 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        for param, value in paramDict.items():
            if line.strip().startswith(param + '='):
                lines[i] = f'{param}={value}\n'
    with open(str(dat_path), 'w') as f:
        f.writelines(lines)

#TA: helper to update ATMOSPH.IN data rows by position; rows is a list of {col_name: value} dicts
def setAtmosphRows(ExpFileLocation, rows):
    atmo_path = ExpFileLocation / 'ATMOSPH.IN'
    with open(str(atmo_path), 'r') as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if 'tAtm' in l.split())
    headers = lines[header_idx].split()
    for row_offset, col_updates in enumerate(rows):
        line_idx = header_idx + 1 + row_offset
        for col_name, value in col_updates.items():
            col_idx = headers.index(col_name)
            lines[line_idx] = writeLine(lines[line_idx], col_idx, value)
    with open(str(atmo_path), 'w') as f:
        f.writelines(lines)

#TA: helper function to update SELECTOR.IN parameters for the current HYDRUS experiment
def setSelectorParams(ExpFileLocation, paramDict, nmat = 1):

    selectorIN = SELECTORIN(ExpFileLocation)

    for param in list(paramDict.keys()):
        selectorIN.setData(param, paramDict[param], mat = nmat)

    selectorIN.update()

#TA: entry point for the script when executed directly
def main():
    run()
             
if __name__ == "__main__":
    main()
       

        
