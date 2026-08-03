#SC_Class.py
#Created by Derek Groenendyk
#11/13/2011
#Updated by Trevor Askins 06/06/2026
#Contains the WC class for water content calculations and soil parameter loading

#TA: import numerical, Python runtime, and MATLAB I/O libraries used by WC utilities
import numpy as np
import sys
import pickle as pkl
from scipy.io import loadmat,savemat


class WC:

    #TA: wrapper class for water content calculations and soil texture parameter loading
    def __init__(self,directory):
        self.name = 'WC'
        self.srcDrive = directory
       

    #TA: compute water content values for a list of pressure heads using provided soil parameters
    def averageWC(self,heads,params):

        wc = []
        for head in heads:
            wc.append(self.calcWC(params,head))

        return wc

    #TA: calculate a single Van Genuchten water content for a given pressure head and soil parameters
    def calcWC(self,params,head):
        
        theta_r = params[0]
        theta_s = params[1]
        alpha = params[2]
        n = params[3]
        K_s = params[4]

        m = 1.0-1.0/n

        vGWC = theta_r + (theta_s - theta_r)/((1.0+(alpha*abs(head))**n)**m)

        return vGWC


    #TA: load soil texture parameter data and return parameter arrays based on user options
    def get_params(self, prct='two', model='old', lmean=True, llog=False, 
                    amy=False, lnan=False, fname='texture_data', num=0, lnotnan=False):

        data_dict, idx = self.get_texture_pkl(prct, fname)

        lens = []
        # for key,value in data_dict['idx'][model+'_one'].items():
        #     lens.append(len(value))
        # print(sorted(lens)[::-1])

        # print(idx)
 
        temp_param_vals = data_dict['params'][model + '_one']

        # print(temp_param_vals.shape)

        idxfin = None

        if model == 'new' and True:
            idxall = data_dict['idx'][model + '_one']

            # print(idxall)
            # print(len(idxall))

            # remove bad soils from each individual soil
            # alist = list(idxall.keys())
            # for key in alist:
            #     if key not in idx:
            #         idxall.pop(key)

            # idxfin = OrderedDict()
            # for key,value in idxall.items():
            #     ind = idx.index(key)
            #     idxfin[ind] = value

            



        # print(idxfin[3])
        # print()

        # print(idxall[6])
        # except (KeyError, AttributeError):
        #     idxall = None

        # print(idxall.keys())

        # print(data_dict.keys())
        # nans = data_dict['idx'][0]
        # print(nans.shape)
        # nanvalues = temp_param_vals[nans[:,0],:,nans[:,1]]
        # print(nanvalues)

        # print(np.nonzero(np.isnan(temp_param_vals[:,4,:]) == True)[0].shape)

        # print(np.isnan(param_vals))

        #TA: apply the new soil model indexing and filter parameter values using the selected index list
        if model == 'new':
            # nans = data_dict['idx'][0]
            # print(temp_param_vals.shape)
            temp_param_vals = temp_param_vals[idx,:,:]

            # print(np.nonzero(temp_param_vals[48, 0, :] == 1.26128206e-01)[0])
            # print(temp_param_vals[48, :, [877]])

            # temp_idx = np.nonzero(temp_param_vals[:,4,:] < 0.006)
            # print(temp_idx)
            # tind = np.nonzero(temp_idx[0] == 48)
            # print(temp_idx[1][tind])
            # print(temp_param_vals[48, :, temp_idx[1][tind]])
            # print(len(np.nonzero(np.array(idxall) < 370)[0]))
            # print(np.nonzero(np.array(idxall) == 369)[0])
            # print(sorted(idxall))

            # print(np.nonzero(temp_param_vals[48, :, :] == [ 0.12614309,  0.57611538,  0.02162044,  1.30832214,  0.00418277]))
        
            if num != 0:
                # print('num',num)
                param_vals = np.zeros((len(idx), 5, num))                

                if False:
                    # remove bad soils from each individual soil
                    for soil in range(len(idx)):
                        # not_nans = np.nonzero(np.isnan(temp_param_vals[soil, 3, :]) == False)[0][:num]
                        try:
                            bad_soils = idxall[idx[soil]]
                            bad_soils = idxfin[soil]
                            # if idx[soil] == 6:
                            #     print(bad_soils)
                        except KeyError:
                            # print('keyerror',idx[soil])
                            bad_soils = []
                        good_soils = np.arange(989)
                        good_soils = np.delete(good_soils,bad_soils)[:num]
                        param_vals[soil, :, :] = temp_param_vals[soil, :, good_soils].T # remove transpose??
                if True:
                    # remove bad soil group all soils
                    good_soils = np.delete(np.arange(989), idxall)[:num]
                    param_vals = temp_param_vals[:, :, good_soils]

            else:
                param_vals = temp_param_vals
                # print(param_vals.shape)

            # print(param_vals[48,:,78])

            if lmean:
                tempvals = np.mean(param_vals, axis=2)
                base = np.ones((param_vals.shape[0],3)) * 10.0
            else:
                tempvals = param_vals
                base = np.ones((param_vals.shape[0],3,param_vals.shape[2])) * 10.0

            if llog:
                # pass
                # unlog values
                # tempvals[:,2:] = np.power(base,tempvals[:,2:])
                # tempvals = np.around(tempvals,4)
                # tempvals[:,-1] = np.around(tempvals[:,-1],2)
                # relog values
                tempvals[:,2:] = np.log10(tempvals[:,2:])
            else:
                tempvals = np.around(tempvals,4)
                tempvals[:,-1] = np.around(tempvals[:,-1],2)

            # print(np.nonzero(tempvals[48, 0, :] == 0.1261)[0])

            param_vals = tempvals
            # print(param_vals.shape)

        #TA: handle the legacy model by slicing parameter arrays and computing statistics separately for Ks and the other parameters
        else:
            temp_param_vals[0] = temp_param_vals[0][idx,:,:]
            temp_param_vals[1] = np.squeeze(temp_param_vals[1][idx,:])

            param_vals = [[], []]
            param_vals[0] = np.zeros((len(idx), 4, num))
            param_vals[1] = np.zeros((len(idx), num))

            if num != 0:
                for soil in range(len(idx)):
                    # print(temp_param_vals[0][soil, 3, :])
                    # not_nans = np.nonzero(np.isnan(temp_param_vals[0][soil, 3, :]) == False)[0][:num]
                    param_vals[0][soil, :, :] = temp_param_vals[0][soil, :, :num]
                    # not_nans = np.nonzero(np.isnan(temp_param_vals[1][soil, :]) == False)[0][:num]
                    param_vals[1][soil, :] = temp_param_vals[1][soil, :num]
            else:
                param_vals[0] = temp_param_vals[0]
                param_vals[1] = temp_param_vals[1]

            # param_vals[0] = param_vals[0][idx,:,:]
            # param_vals[1] = np.squeeze(param_vals[1][idx,:,:])
            
            if lmean:
                tempvals = np.mean(param_vals[0],axis=2)
                tempvals_Ks = np.mean(param_vals[1],axis=1)
                base = np.ones((tempvals.shape[0],2)) * 10.0
                base_Ks = np.ones((tempvals_Ks.shape[0])) * 10.0
                tempParams = np.zeros((param_vals[0].shape[0],5))
            else:
                tempvals = param_vals[0]
                tempvals_Ks = param_vals[1]   
                base = np.ones((tempvals.shape[0],2,tempvals.shape[2])) * 10.0
                base_Ks = np.ones((tempvals_Ks.shape[0],tempvals_Ks.shape[1])) * 10.0
                tempParams = []       

            if llog:
                # unlog values
                # tempvals[:,2:] = np.power(base,tempvals[:,2:])
                # tempvals_Ks[:] = np.power(base_Ks,tempvals_Ks[:])
                # tempvals = np.around(tempvals,4)
                # tempvals_Ks = np.around(tempvals_Ks,2)

                # relog values
                tempvals[:,2:] = np.log10(tempvals[:,2:])
                tempvals_Ks[:] = np.log10(tempvals_Ks[:])
            else:
                tempvals = np.around(tempvals,4)
                tempvals_Ks = np.around(tempvals_Ks,2)

            if lmean:
                tempParams[:,:4] = tempvals
                tempParams[:,4] = tempvals_Ks
            else:
                tempParams.append(tempvals)
                tempParams.append(tempvals_Ks)

            param_vals = tempParams
            

        #TA: optionally override parameter values by reading only Van Genuchten data from text input
        if amy:
            infile = open(str(self.srcDrive  / 'soil_texture_data' / 'onlyVanGenuchten.txt'),'r')
            lines = infile.readlines()
            infile.close()
            param_vals = np.zeros((1326,6))
            i = -1
            for line in lines:
                i += 1
                data = line.split()
                for j in range(5):
                    if j == 2:
                        # round and convert Alpha, to cm/day from m/s
                        temp = round(float(data[j+1])/100.0,4)
                    elif j == 4:
                        # round and convert Ks, to cm/day from m/s
                        temp = round((float(data[j+1])*100.0)*(60*60*24),2)
                    else:
                        temp = round(float(data[j+1]),4)
                    # temp = data[j+1]  
                    param_vals[i,j] = temp
            param_vals[:,5] = np.log10(param_vals[:,4])                     
 

        # print(param_vals[78,:,47:51])
        # print(param_vals[47:51,:,78])

        if not lnotnan:
            return param_vals
        else:
            return param_vals, idxfin

    #TA: load and decode soil texture data from pickle files, optionally returning a filtered index list
    def get_texture_pkl(self, prct, fname='texture_data'):
        # pkl_file = open('E:\\Derek\\ProgrammingFolder\\Soil Texture Data\\' + fname + '.pkl','rb')
        pkl_file = open(str(self.srcDrive / 'soil_texture_data' / (fname + '.pkl')),'rb')

        if sys.version_info[0] == 3:
            # print("version 3")
            data_dict = pkl.load(pkl_file, fix_imports=True, encoding="bytes", errors="strict")

            # print(SSC_Data[48:52])
            # print(SSC_Data[98:102])       
            
            for key in list(data_dict.keys()):
                # print(data_dict[key])
                # print(key)
                new_key = key.decode('utf-8')
                data_dict[new_key] = data_dict.pop(key)
                for akey in list(data_dict[new_key].keys()):
                    new_akey = akey.decode('utf-8')
                    data_dict[new_key][new_akey] = data_dict[new_key].pop(akey)

            # print(data_dict['fractions']['one_prct'][[96,98,100,201]])
            # print(data_dict['fractions']['one_prct'][[295,297,299,398]])                    

        else:
            # print('version 2')
            data_dict = pkl.load(pkl_file)
        pkl_file.close()

        # pkl_file = open('C:\\Derek\\ProgrammingFolder\\Soil Texture Data\\texture_data.pkl','wb')
        # pkl.dump(data_dict, pkl_file, protocol=2)
        # pkl_file.close()

        idx = list(np.arange(5151))
        if prct == 'two':
            # pkl_file = open('E:\\Derek\\ProgrammingFolder\\Soil Texture Data\\idx_twoprct.pkl','rb')
            pkl_file = open(str(self.srcDrive / 'soil_texture_data/idx_twoprct.pkl'),'rb')

            if sys.version_info[0] == 3:
                idx = pkl.load(pkl_file,fix_imports=True,encoding="bytes", errors="strict")
            else:
                idx = pkl.load(pkl_file)
            pkl_file.close()

            # pkl_file = open('C:\\Derek\\ProgrammingFolder\\Soil Texture Data\\idx_twoprct.pkl','wb')
            # pkl.dump(idx, pkl_file, protocol=2)
            # pkl_file.close()

        elif prct == 'five_twenty':
            # 5% intervals, all components (sand, silt, clay) >= 20%
            # Ordering of 5151 soils: index(sand, silt) = 101*sand - sand*(sand-1)//2 + silt
            idx = []
            for sand in range(20, 65, 5):
                for silt in range(20, 81 - sand, 5):
                    idx.append(101 * sand - sand * (sand - 1) // 2 + silt)

        return data_dict, idx
 
    #TA: retrieve soil saturation conductivity fractions for the requested percentage index selection
    def getSSC(self, prct='two'):
        data_dict, idx = self.get_texture_pkl(prct)

        SSC_Data = data_dict['fractions']['one_prct'][idx]

        # print(idx[48:52])
        # print(idx[98:102])

        # print(SSC_Data[48:52])
        # print(SSC_Data[98:102])

        # if prct == 'one':
        #     infile = open(self.srcDrive+'ProgrammingFolder\\Soil Texture Data\\SSC_1percent.txt','r')
        # elif prct == 'two':
        #     infile = open(self.srcDrive+'ProgrammingFolder\\Soil Texture Data\\SSC.txt','r')
        # lines = infile.readlines()
        # infile.close()
        # mdata = [[float(s.split()[i+1]) for i in range(3)] for s in lines[1:len(lines)]]
        # SSC_Data = np.array(mdata,dtype=np.float)

        # infile = open(self.srcDrive+'ProgrammingFolder\\newSoilTex.txt','r')
        # lines = infile.readlines()
        # infile.close()

        # ssc = np.zeros((1326,3))

        # i = -1
        # for line in lines:
        #     i += 1
        #     data = line.split()
        #     for j in range(3):
        #         temp = data[j+1]
                                 
        #         ssc[i,j] = temp

        return SSC_Data


    #TA: plot water content versus time using matplotlib for quick visualization
    def plotWC(self, wc):

        fig=plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(wc)
        ax.set_xlabel('Days')
        ax.set_ylabel('$\\theta$, cm$^3$/cm$^3$')

        ax.set_xticklabels(['1','2','3','4','5'])
       
    ##    ax.set_ylim((0,0.025))
    ##    ax.set_xlim((0,0.2))
        plt.show()

    #TA: compute field capacity water content from soil parameters using empirical scaling
    def fc(self, params):
        s = params[3]**(-.6 * (2. + np.log10(params[4])))
        thfc = s * (params[1] - params[0]) + params[0]

        return thfc


    #TA: compute water content for a given suction using the Van Genuchten model and a soil parameter vector
    def vanGsm(self, params, psi):

        # paramList = ['thr','ths','Alfa','n','Ks']
       
        m = 1.0 - (1.0 / params[3])

        theta = lambda h:  params[0] + (params[1] - params[0]) * (1.0 / (1.0 + ((params[2] * np.abs(h))**params[3])))**m

        wc = theta(psi)

        return wc

    #TA: invert the Van Genuchten equation to compute pressure head from water content
    def vanGpsi(self,params,wc):

        # paramList = ['thr','ths','Alfa','n','Ks']
       
        m = 1.0-(1.0/params[3])

        psi = lambda wc:  (((((wc - params[0]) / (params[1] - params[0]))**(-1./m)) - 1.)**(1./params[3])) / params[2]

        h = psi(wc)

        return h

    #TA: compute unsaturated hydraulic conductivity using the Mualem-van Genuchten formulation
    def vGMKs(self,params,psi):

        m = 1.0-(1.0/params[3])

        a = 1.0+(params[2]*psi)**params[3]
        b = 1.0-(params[2]*psi)**(params[3]-1)

        Ks = params[4]*((b*(a)**-m)**2.0)/(a**(m/2.0))

        return Ks
   

    #TA: compute unsaturated hydraulic conductivity as a function of water content using van Genuchten parameters
    def vGKs(self,params,theta):

        m = 1.0-(1.0/params[3])

        a = (theta-params[0])/(params[1]-params[0]) # theta must be larger than theta_residual (thr)

        Ks = params[4]*(a**0.5)*((1.0-(1.0-a**(1.0/m))**m)**2.0)

        return Ks

    #TA: compute hydraulic conductivity relative to saturation for a given saturation fraction
    def vGKsSat(self,params,S):

        m = 1.0-(1.0/params[3])

        a = S

        Ks = params[4]*(a**0.5)*((1.0-(1.0-a**(1.0/m))**m)**2.0)

        return Ks
        

#TA: module test entry point that can run basic WC parameter loading when executed directly
def main():

   # srcDrive = "E:\\"
   
   # exp = "SW605"
   
   # ExpFileLocation = srcDrive+"ProgrammingFolder\\HYDRUS_Data\\Projects\\"+exp

   # NodInfOUT = NODINF(ExpFileLocation)
   # selectorIN = SELECTORIN(ExpFileLocation)

   # paramList = ['thr','ths','Alfa','n','Ks']
   # paramData = []
   
   # for param in paramList:
   #     paramData.append(float(selectorIN.getData(param)))

   # headData = NodInfOUT.getAvgHeadData(50)
   # print np.shape(headData)

   # wc = averageWC(headData,paramData)

   # plotWC(wc)


    wc = WC()

    p,idx = wc.get_params('two', 'new', lmean=False, num=698, lnotnan=True)

    # print(p.shape)
    # print(idx[0][131])

    # print(p[0,:,125:135])

    # print(p[0,:,0])
    # p = [0.145, 0.4826, 0.0073, 1.3291, 16.99]
    # p = [0.034, 0.46, 0.016, 1.37, 7.8]
    # print(wc.vanGsm(p,330.))
    # print(wc.fc(p))

    # print(idx.keys())
    # for key in idx.keys():
    #     print(idx[1])

    # p,_ = wc.get_params('one', 'new', False, num=20, lnotnan=True, fname='texture_data_2')

    # print(p[0,:,idx[0][0]])




   # p = p[1]

   # print(p.shape)
   # print(len(np.nonzero(np.isnan(p))[0]))

   # print(p[0,:,0])

if __name__=='__main__':
   main()






























    

        
