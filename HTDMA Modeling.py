# read in libraries
import sys, os
import netCDF4
import scipy.io, numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

directory = '/Users/evenhou/Downloads/armdata'
# kappa_50 = pd.Series() # array, for all kappa in all file with size = 50 nm
# conc_norm1_50 = pd.Series() # array, for all normalized conc in all file with size = 50 nm
# #volume fraction at bin i
# p_LH_i = pd.Series()
# p_MH_i = pd.Series()
kappa_LH = 0.01
kappa_MH = 0.6

for subdir, dirs, files in os.walk(directory):
    for file in files:
        filename = subdir + os.sep + file
        if ('sgpaoshtdmaE13.b1.20210427' in filename) and ('.nc' in filename):
# =============== FOR EACH FILE ===============
# get dimensions of time and bins
            ds = netCDF4.Dataset(filename)
            s1 = str(ds.dimensions['time'])
            s2 = str(ds.dimensions['bin'])
            # print(s)
            x = -3
            if (s1[x-1:x+1] != '='):
                dim_time = int(s1[x:])
            else:
                dim_time = int(s1[x-1:])
            if (s2[x-1:x+1] != '='):
                dim_bin = int(s2[x:])
            else:
                dim_bin = int(s2[x-1:])
            # print(dim_time, dim_bin)
# CHANGE HERE: define arrays
            time = np.zeros(dim_time)
            dry_dia = np.zeros(dim_time)
            kappa = np.zeros((dim_time, dim_bin))
            k_bound = np.zeros(((dim_time, dim_bin,2)))
            conc = np.zeros((dim_time, dim_bin))
            # print(time.shape, dry_diam.shape)
            
# CHANGE HERE: read data into arrays
            ncf = scipy.io.netcdf_file(filename, mmap=False)
            date = int(filename[-18:-10]) # read i.e. 20210427
            # print(date)
            time[:] = ncf.variables["time"].data/3600 # hr
            dry_dia[:] = ncf.variables["dry_diameter_setting"].data # nm
            for i_time in range(dim_time):
                kappa[i_time, :] = ncf.variables['kappa'][i_time, :]
                k_bound[i_time, :, :] = ncf.variables['kappa_bounds'][i_time, :, :]
                conc[i_time,:] = ncf.variables['aerosol_concentration'][i_time, :] # dN, unit: 1/cm^3
# calculations
            dK = np.zeros((dim_time,dim_bin))
            for i_time in range(dim_time):
                dK[i_time, :] = k_bound[i_time, :, 1] -  k_bound[i_time, :, 0]
            conc_norm0 = np.zeros((dim_time,dim_bin)) # dN/dK
            for i_time in range(dim_time):
                conc_norm0[i_time, :] = conc[i_time, :]/dK[i_time, :]
            conc_norm1 = np.zeros((dim_time,dim_bin)) # dN/dK * 1/N_tot = dN°/dK (normalized)
# use pandas to manipulate data
            d = {'dry_dia': dry_dia, 'time':time}
            df = pd.DataFrame(data = d)
            #print(df)
    
            df_conc = pd.DataFrame(conc)
            #print(df_conc)
            N_tot = df_conc.sum(axis=1)# N_tot (series) for each scan
            N_tot.name = "N_tot"
            #print(N_tot) 
            #print(type(N_tot)) # Series
            
            df = df.join(N_tot)
            #print(df)
            
            newdf = pd.DataFrame(np.repeat(df.values, dim_bin, axis=0),
                                 columns=df.columns)
            #print(newdf.shape)
            newdf['kappa'] = kappa.ravel() # K_i
            newdf['conc'] = conc.ravel()
            newdf['dK'] = dK.ravel()
            newdf['conc_norm0'] = conc_norm0.ravel()  # dN/dK
            newdf['conc_norm'] = newdf['conc_norm0']/newdf['N_tot'] # dN°/dK = c(k)_i
            newdf['p_MH_i'] = (newdf.kappa-kappa_LH)/(kappa_MH-kappa_LH)
            newdf['p_LH_i'] = 1 - newdf['p_MH_i']
            #print(newdf)
    
            newdf_valid = newdf[(newdf['kappa'] >= 0.01) & (newdf['kappa'] <= 0.6)]
            #print(newdf_valid.head(60))
            
            # change N_tot from pd Series to array
            N_tot = N_tot.array
            #print(type(N_tot))
            
            N_tot_new = newdf_valid.N_tot # type: series
            N_tot_new = newdf_valid['N_tot'].to_numpy()
            N_tot_new = N_tot_new.tolist()
            #print(type(N_tot_new))
            
            count = []
            ct = 0
            for i in range(0,len(N_tot)):
                ct = N_tot_new.count(N_tot[i]) # count the amount of appearance of N_tot[i] in N_tot_new
                count = np.append(count,ct)
                        
            #print(count)
            #print(type(count)) : np array
            #print(len(count))
            
            print(newdf_valid.head(60))
            
            # for each file seperately
            conc_norm = newdf_valid.conc_norm # c(k)_i
        
            p_MH_i = newdf_valid.p_MH_i
            p_LH_i = newdf_valid.p_LH_i
            
            dK = newdf_valid.dK
            
            # mixing entropy for particle at bin i
            H_i = -p_LH_i*np.log(p_LH_i)-p_MH_i*np.log(p_MH_i)
            H_i.name = "H_i"
            newdf = newdf_valid.join(H_i)
            

            # average mixing entropy: H_alpha
            H_alpha_i = H_i*(conc_norm/N_tot)*dK
            
            p_MH_j = p_MH_i*(conc_norm/N_tot)*dK
            p_LH_j = p_LH_i*(conc_norm/N_tot)*dK
            
            # print(H_alpha_i.head(20))
            # print(H_alpha_i[18]) # [index]
            H_alpha_i.name = 'H_alpha_i'
            p_MH_j.name = 'p_MH_j'
            p_LH_j.name = 'p_LH_j'
            newdf = newdf.join(H_alpha_i)
            newdf = newdf.join(p_MH_j)
            newdf = newdf.join(p_LH_j)
            #print(newdf)
            
            # change NaN value to 0
            newdf['H_alpha_i'] = newdf['H_alpha_i'].fillna(0)
            print(newdf)
            #print(newdf.H_alpha_i.head(20))
            
            # transfer from pd series to pd array
            H_alpha_i = newdf.H_alpha_i.array 
            p_MH_j = newdf.p_MH_j.array
            p_LH_j = newdf.p_LH_j.array
            kappa = newdf.kappa.array
            
            
            x = 0
            end = int(len(H_alpha_i)/60) # 60 in a group
            #print(end) # 56
            
            H_alpha = [] # empty array
            p_MH = []
            p_LH = []
            kappa_avg = []
            
            for i in range(0,end-1): # first n-1 groups, assuming there are n groups # range(0,55)-> 0~54
                sum_H_alpha_i = sum(H_alpha_i[range(x,x+60)])
                sum_MH_i = sum(p_MH_j[range(x,x+60)]) 
                sum_LH_i = sum(p_LH_j[range(x,x+60)]) 
                k_avg = sum(kappa[range(x,x+60)])/60
                #print(sum_MH_i)
                H_alpha = np.append(H_alpha,sum_H_alpha_i)
                p_MH = np.append(p_MH,sum_MH_i)
                p_LH = np.append(p_LH,sum_LH_i)
                kappa_avg = np.append(kappa_avg,k_avg)
                x = x+60
            #print(p_MH)
            #print(len(p_LH))
                
            # last group (nth group)
            sum_H_alpha_i_last = sum(H_alpha_i[-60:])
            sum_MH_i_last = sum(p_MH_j[-60:])
            sum_LH_i_last = sum(p_LH_j[-60:])
            k_avg_last = sum(kappa[-60:])/60
        
            H_alpha = np.append(H_alpha,sum_H_alpha_i_last) # for each scan, across all bins
            p_MH = np.append(p_MH,sum_MH_i_last)
            p_LH = np.append(p_LH,sum_LH_i_last)
            kappa_avg = np.append(kappa_avg,k_avg_last)
            
            # bulk mixing entropy: H_gamma for each scan
            H_gamma = -p_LH*np.log(p_LH)-p_MH*np.log(p_MH)
            #print(H_alpha)
            #print(type(H_gamma))
        
            # diversity: D for each scan
            D_i = np.exp(H_i)
            D_alpha = np.exp(H_alpha)
            D_gamma = np.exp(H_gamma)
            
            # hygroscopic heterogeneity parameter
            chi = (D_alpha-1)/(D_gamma-1)
            #print(chi)
            
            # change array to pd series
            D_alpha = pd.Series(D_alpha)
            D_gamma = pd.Series(D_gamma)
            chi = pd.Series(chi)
            kappa_avg = pd.Series(kappa_avg)
            D_alpha.name = 'D_alpha'
            D_gamma.name = 'D_gamma'
            chi.name = 'chi'
            kappa_avg.name = 'kappa_avg'
            
            df_Fig3 = pd.DataFrame(D_alpha)
            #print(df_Fig3)
            
            df_Fig3 = df_Fig3.join(D_gamma)
            df_Fig3 = df_Fig3.join(chi)
            df_Fig3 = df_Fig3.join(kappa_avg)
            #print(df_Fig3)
            
            #df_Fig3['D_gamma'] = df_Fig3['D_gamma'].fillna(0)
            #df_Fig3['chi'] = df_Fig3['chi'].fillna(0)
            #print(df_Fig3.head(50))