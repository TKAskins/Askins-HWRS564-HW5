## Full_reprocessData.py
## Derek Groenendyk
## 9/17/2013
## for clustering data


import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.io import loadmat,savemat

import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.font_manager import fontManager, FontProperties
from matplotlib.ticker import MultipleLocator,AutoMinorLocator
import matplotlib.colors as colors
import matplotlib.cm as cmx

##   End of Imports
#################################################################################
#################################################################################
##  Start of Code

currentDir = Path(os.getcwd())

    
def saveFig(fig,filename,ext,DPI=300):
    fig.savefig(str(currentDir / 'images' / (filename + '.' + ext)), format = ext, dpi = DPI)            


def importSSCData(directory):
    
    infile = open(str(directory / 'soil_texture_data' / 'SSC.txt'), 'r')
    lines = infile.readlines()
    infile.close()
    mdata = [[float(s.split()[i+1]) for i in range(3)] for s in lines[1:len(lines)]]
    SSC_Data = np.array(mdata)
    
    return SSC_Data

def findSoilIndex(ssc):

    sand = ssc[0]
    silt = ssc[1]
    clay = ssc[2]

    SSC_Data = importSSCData()
    for i in range(1326):
        if SSC_Data[i,0] == sand:
            for j in range(i,1326):
                if SSC_Data[j,0] == sand and \
                   SSC_Data[j,1] == silt:
                    for k in range(j,1326):
                        if SSC_Data[k,0] == sand and \
                           SSC_Data[k,1] == silt and \
                           SSC_Data[k,2] == clay:
                            index = k
    try:
        return index
    except UnboundLocalError:
        print(ssc)
##    print SSC_Data[index]


def get_hourly_data(directory, plant_type=None):

    directory = Path(directory)
    all_mat_files = sorted(directory.glob('Trial*.mat'))

    # detect PlantType from the last segment of the first trial name (e.g. "...rd50.corn")
    # if plant_type is specified, filter to only that plant type
    if plant_type is None:
        first_trial_name = all_mat_files[0].stem[len('Trial'):]
        plant_type = first_trial_name.split('.')[-1]

    mat_files = [f for f in all_mat_files if f.stem.split('.')[-1] == plant_type]
    print(f'Processing {len(mat_files)} {plant_type} files')

    first_mat = loadmat(str(mat_files[0]))['data']
    times = np.squeeze(first_mat['t'][0][0][0])

    # auto-detect node spacing and build depth index/values arrays
    num_nodes = first_mat['wc'][0][0].shape[1]
    node_spacing_cm = 200 / (num_nodes - 1)
    node_step = max(1, int(round(5 / node_spacing_cm)))  # nodes per 5 cm
    node_indices = np.arange(0, num_nodes, node_step)
    depths = (node_indices * node_spacing_cm).astype(int)  # actual depths in cm

    # detect time resolution: sub-daily steps → hourly labels, else daily
    dt = float(np.median(np.diff(times)))
    if dt < 0.1:
        t_unit = 'hours'
        t_labels = [int(round(t * 24)) for t in times]
    else:
        t_unit = 'days'
        t_labels = [int(round(t)) for t in times]

    tidx = list(range(len(times)))

    index = pd.MultiIndex.from_product([t_labels, depths], names=[t_unit, 'd'])

    df = None

    for mat_path in mat_files:
        trial_name = mat_path.stem[len('Trial'):]  # strip 'Trial' prefix
        print('reading trial: ', trial_name)

        mat_file = loadmat(str(mat_path))['data']
        wc_tempdata = mat_file['wc'][0][0]
        h_tempdata  = mat_file['p'][0][0]
        f_tempdata  = mat_file['f'][0][0]

        cindex = pd.MultiIndex.from_product(
            [['wc', 'p', 'f'], [trial_name]], names=['type', 'trial'])

        trial_df = pd.DataFrame(
            np.array([wc_tempdata[tidx][:, node_indices].flatten(),
                      h_tempdata[tidx][:, node_indices].flatten(),
                      f_tempdata[tidx][:, node_indices].flatten()]).T,
            index=index, columns=cindex)

        df = trial_df if df is None else pd.concat([df, trial_df], axis=1)

    return df, t_unit, plant_type


def plot_hourly_data(directory, csv_fname, fname):

    # SSC_Data = importSSCData()
    # print(SSC_Data[343])
    # raise    

    df = pd.read_csv(str(directory / csv_fname), header=[0,1], index_col=[0,1])    

    idx = pd.IndexSlice

    soils = {
    'sandy loam':[82,18,0],
    'loam':[54,10,36],
    'silt':[54,40,6],
    'silt loam':[50,26,24],
    'clay loam':[26,26,48],
    'clay':[34,44,22]}

    soils = {
    'clay2':[14,14,72]}

    soils = [579,343,660,1301,985,1268]

    soils = [343]
    
    tidx = range(1,24,4) 

    depths = range(-5,-55,-5)

    # for soil_ind in [250,345,550,745,890,1100]:

    #     df.loc[idx[:, 20], ('wc', soil_ind)].plot()
    #     # df.loc[idx[12, :], ('wc', soil_ind)].plot()

    # for soil,ssc in soils.items():
    #     soil_ind = findSoilIndex(ssc)
    for ind in soils:
        soil_ind = ind
        soil = str(ind)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i,t in enumerate(tidx):

            data = df.loc[(idx[t, :]), ('wc', soil)]
            # print(list(depths))

            ax.plot(data, depths, label='t=' + str(t) + ' hrs')

        ax.set_title('Soil #: ' + soil + ' - ' + fname)
        ax.set_ylabel('depth, cm')
        ax.set_xlabel('$\\theta$')

        ax.set_ylim([-50,0])
        ax.set_xlim([0.0,0.6])

        handles, labels = ax.get_legend_handles_labels()
        font= FontProperties(size=12)
        leg = ax.legend(handles, labels,loc='lower left',prop=font,numpoints=1)        

        fname = soil + '_' + fname

        saveFig(fig, fname, 'png')        



def plot_raw_data(directory, exp):

    os.chdir(str(directory))

    # trial = 745
    # sInd = findSoilIndex([34,32,34])
    # print(sInd)
    # raise

    soils = {
    'sand':[94,6,0],
    'loamy sand':[86,14,0],
    'sandy loam':[82,18,0],
    'loam':[54,10,36],
    'silt':[54,40,6],
    'silt loam':[50,26,24],
    'sandy clay loam':[44,2,54],
    'clay loam':[26,26,48],
    'silty clay loam':[20,44,36],
    'sandy clay':[32,0,68],
    'silty clay':[26,52,22],
    'clay':[34,44,22]}

    soils = {
    'sandy loam':[82,18,0],
    'loam':[54,10,36],
    'silt':[54,40,6],
    'silt loam':[50,26,24],
    'clay loam':[26,26,48],
    'clay':[34,44,22]}

    soils = [579,343,660,1301,985,1268]

    soils = [343]

    # for soil,ssc in soils.items():
    #     print(soil, findSoilIndex(ssc))
    # print(wc_tempdata.shape)

    times = np.squeeze(loadmat('Trial' + str(soils[0]) + '.0')['data']['t'][0][0][0])

    tidx = []
    for t in range(1,24,4):
        ind = np.argmin(np.absolute(t - times))
        tidx.append(ind)

    tidx = np.array(tidx).astype(int)

    # print(times)

    # print(tidx)
    # raise

    # tidx = [0, -1]
    # tidx = [0, 20, 40]

    depths = range(-1,-202,-1)

    # for soil,ssc in soils.items():
    #     soil_ind = findSoilIndex(ssc)
    for ind in soils:
        soil_ind = ind
        soil = str(ind)

        wc_tempdata = loadmat('Trial' + str(soil_ind) + '.0')['data']['wc'][0][0]

        # print(wc_tempdata.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i,t in enumerate(tidx):
            data = wc_tempdata[t, :]
            ax.plot(data, depths, label='t=' + str(t) + ' hrs')

        ax.set_title(soil)
        ax.set_ylabel('depth, cm')
        ax.set_xlabel('$\\theta$')

        ax.set_ylim([-200,0])
        ax.set_xlim([0,0.6])

        handles, labels = ax.get_legend_handles_labels()
        font= FontProperties(size=12)
        leg = ax.legend(handles, labels,loc='lower left',prop=font,numpoints=1)        

        fname = soil + '_' + exp

        saveFig(fig, fname, 'png')


def main():

    # set global label sizes in pt. values
    pt = 10
    params = {'backend': 'ps',
              'axes.labelsize':  pt,
              'xtick.labelsize': pt,
              'ytick.labelsize': pt,
              'axes.labelsize':  pt,
              'axes.titlesize':  pt}
    mp.rcParams.update(params)

    exp = 'Flood'
    results_folder = 'daily'

    results_dir = currentDir / 'results' / exp / results_folder

    ###### save CSV file #####
    # specify plant_type to avoid mixing corn and beans results from the same folder
    plant_type = 'beans'
    data_df, t_unit, plant_type = get_hourly_data(results_dir, plant_type=plant_type)
    outfile_fname = f'{exp}_{plant_type}_{t_unit}.csv'
    data_df.to_csv(str(currentDir / outfile_fname))



    ##### plotting functions #####
    # base_fname = exp + '_' + expType + '_raw'
    # plot_raw_data(results_dir, base_fname)


    # csv_fname = exp + '_' + expType + '.csv'
    # base_fname = exp + '_' + expType
    # plot_hourly_data(currentDir, csv_fname, base_fname)

    # csv_fname = exp + '_' + expType + '.csv'
    # df = pd.read_csv(csv_fname, header=[0,1], index_col=[0,1])
    # print(df)




if __name__=='__main__':
    main()
