#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import FittingWorkCleaner as FWC
import Neutrino_Work as NW
from classy import Class


# The function below takes inputs m_small, normal, filename, plot,file,Pk_graphs. m_small is the smallest neutrino mass and should be entered in eV's  here. normal should be true of false. True if the normal hiearachy is wanted and false is the inverted hiearachy is wanted. the filename input requires a filename as an input so that the function can load the e,f, array and other important numbers. Plot is true if you wish to graph the neutrino spectra and false is you do not want to plot the neutrino spectra. file is an input which requires an input in quotes which will be the name of the file saved. Pk_graphs is true if you wish to see the two matter power spectra plots and false if you do not want the graphs plotted when running the function. 

# In[13]:


def v_masses_nontherm_alpha(m_small, normal,filename,plot,file,Pk_graphs):
    
    #if normal is True, uses the normal hiearachy, If normal False, uses the inverted hiearachy.
    #if plot is true, neutrino spectra is plotted. If false, spectra is not plotted. parameters returned in both 
    #if Pk_plots is true, then the matter power spectra plots are plotted. 
    
    actual_data= np.load(filename, allow_pickle=True)
    f_array = actual_data['fe'][-1]
    e_array = actual_data['e'][-1]
    alpha = 1
    poly_degree = 4
    
    mass1,mass2,mass3 = NW.v_masses(m_small,normal)
    
    T_best,N_best,coefficients = FWC.finale(e_array,f_array,poly_degree,plot)
    
    if coefficients[0]>0:
        A = 0 
        
    else:
        A = coefficients[0]
    
    params = '{},{},{},{},{},{},{}'.format(T_best/alpha,N_best,A*alpha**4,coefficients[1]*alpha**3,coefficients[2]*alpha**2,coefficients[3]*alpha,coefficients[4])
    
    af = actual_data['scalefactors'][-1]
    tf = actual_data['temp'][-1]
    value = (1/(af*tf))*alpha

    
    neutrino_mass_settings = {'N_ncdm':3,
          'use_ncdm_psd_files': "0,0,0",
          'm_ncdm': '{},{},{}'.format(mass1,mass2,mass3),
          'T_ncdm':'{},{},{}'.format(value,value,value),
          'ncdm_psd_parameters': params,
          'N_ur': 0.0
          }   
    
    LambdaCDM_settings = {'omega_b':0.0223828,
                     #'omega_cdm':0.1201075,
                     'h':0.67810,
                     'A_s':2.100549e-09,
                     'n_s':0.9660499,
                     'tau_reio':0.05430842,
                     'output':'mPk',
                     'P_k_max_1/Mpc':3.0,
                      'Omega_m':0.309883043,
                     # The next line should be uncommented for higher precision (but significantly slower running)
                     'ncdm_fluid_approximation':3,
                     # You may uncomment this line to get more info on the ncdm sector from Class:
                     'background_verbose':1,
                    }

    neutrino = Class()
    neutrino.set(LambdaCDM_settings)
    neutrino.set(neutrino_mass_settings)
    neutrino.compute()

    neutrino_results = NW.dict_results(neutrino)
    N_eff = neutrino_results['Neff']   
        
    LambdaCDM_other = {'N_ur':'{},{},{}'.format(N_eff,N_eff,N_eff) #needed to get LambdaCDM Neff closer to model Neff
                     }
    
    LambdaCDM = Class()
    LambdaCDM.set(LambdaCDM_settings)
    LambdaCDM.set(LambdaCDM_other)
    LambdaCDM.compute()
    
    LambdaCDM_results = NW.dict_results(LambdaCDM)
    
    kk = np.logspace(-4,np.log10(3),1000) # k in h/Mpc
    Pk_LambdaCDM = np.zeros(len(kk)) # P(k) in (Mpc/h)**3
    h = LambdaCDM_settings['h'] # get reduced Hubble for conversions to 1/Mpc

    for i,k in enumerate(kk):
        Pk_LambdaCDM[i] = LambdaCDM.pk(k*h,0.)*h**3 # function .pk(k,z)

    
    kk= np.logspace(-4,np.log10(3),1000) # k in h/Mpc
    Pk_neutrino = np.zeros(len(kk))

    for i,k in enumerate(kk):
        Pk_neutrino[i] = neutrino.pk(k*h,0.)*h**3 # function .pk(k,z)
        
        
    np.savez(file, n_results = neutrino_results, L_results = LambdaCDM_results, k_n_array = kk, Pk_n_array = Pk_neutrino, LambdaCDM_array = Pk_LambdaCDM, truth_value = normal, v1 = mass1, v2 = mass2, v3 = mass3, small_mass = m_small, sum_n = mass1+mass2+mass3, alpha_value = alpha)

        
    if Pk_graphs:
        plt.figure()
        plt.loglog(kk,Pk_LambdaCDM)
        plt.loglog(kk,Pk_neutrino,linestyle='--')
        plt.xlabel(r'$k \,\,\,\, [h/\mathrm{Mpc}]$')
        plt.ylabel(r'$P(k) \,\,\,\, [\mathrm{Mpc}/h]^3$')
        plt.show()


        plt.figure()
        plt.semilogx(kk,Pk_neutrino/Pk_LambdaCDM-1)
        plt.xlabel(r'$k [h /\mathrm{Mpc}]$')
        plt.ylabel(r'$P(k)^\nu/P(k)-1$')
        plt.show()
        
        return neutrino_results,LambdaCDM_results
    
    else: 
        return neutrino_results,LambdaCDM_results
        


# The function below takes input m_small (a number) in eV. normal follows the same thing as the function above. saved_file requires an input in quotes and is the name of the saved file. power_graphs works the same as Pk_graphs mentioned in the previous function. 

# In[14]:


def v_masses_therm(m_small, normal,saved_file,power_graphs):
    
    #if normal is True, uses the normal hiearachy, If normal False, uses the inverted hiearachy.
    
    mass1,mass2,mass3 = NW.v_masses(m_small,normal)
    
    T,N,A,B,C,D,E = 1,1,0,0,0,0,-100
    
    params = '{},{},{},{},{},{},{}'.format(T,N,A,B,C,D,E)
    
    
    neutrino_mass_settings = {'N_ncdm':3,
          'use_ncdm_psd_files': "0,0,0",
          'm_ncdm': '{},{},{}'.format(mass1,mass2,mass3),
          'T_ncdm': "0.71,0.71,0.71",
          'ncdm_psd_parameters': params,
          'N_ur': 0.0
          }   
    
    LambdaCDM_settings = {'omega_b':0.0223828,
                     #'omega_cdm':0.1201075,
                     'h':0.67810,
                     'A_s':2.100549e-09,
                     'n_s':0.9660499,
                     'tau_reio':0.05430842,
                     'output':'mPk',
                     'P_k_max_1/Mpc':3.0,
                      'Omega_m':0.309883043,
                     # The next line should be uncommented for higher precision (but significantly slower running)
                     'ncdm_fluid_approximation':3,
                     # You may uncomment this line to get more info on the ncdm sector from Class:
                     'background_verbose':1,
                    }

    neutrino = Class()
    neutrino.set(LambdaCDM_settings)
    neutrino.set(neutrino_mass_settings)
    neutrino.compute()

    neutrino_results = NW.dict_results(neutrino)
    N_eff = neutrino_results['Neff']   
        
    LambdaCDM_other = {'N_ur':'{},{},{}'.format(N_eff,N_eff,N_eff) #needed to get LambdaCDM Neff closer to model Neff
                     }
    
    LambdaCDM = Class()
    LambdaCDM.set(LambdaCDM_settings)
    LambdaCDM.set(LambdaCDM_other)
    LambdaCDM.compute()
    
    LambdaCDM_results = NW.dict_results(LambdaCDM)
    
    kk = np.logspace(-4,np.log10(3),1000) # k in h/Mpc
    Pk_LambdaCDM = np.zeros(len(kk)) # P(k) in (Mpc/h)**3
    h = LambdaCDM_settings['h'] # get reduced Hubble for conversions to 1/Mpc

    for i,k in enumerate(kk):
        Pk_LambdaCDM[i] = LambdaCDM.pk(k*h,0.)*h**3 # function .pk(k,z)

    
    kk= np.logspace(-4,np.log10(3),1000) # k in h/Mpc
    Pk_neutrino = np.zeros(len(kk))

    for i,k in enumerate(kk):
        Pk_neutrino[i] = neutrino.pk(k*h,0.)*h**3 # function .pk(k,z)
        
        
    np.savez(saved_file, n_results = neutrino_results, k_n_array = kk, Pk_n_array = Pk_neutrino,Lambda_array = Pk_LambdaCDM, truth_value = normal, v1 = mass1, v2 = mass2, v3 = mass3, small_mass = m_small, sum_n = mass1+mass2+mass3)
        
    if power_graphs:
        
        plt.figure()
        plt.loglog(kk,Pk_LambdaCDM)
        plt.loglog(kk,Pk_neutrino,linestyle='--')
        plt.xlabel(r'$k \,\,\,\, [h/\mathrm{Mpc}]$')
        plt.ylabel(r'$P(k) \,\,\,\, [\mathrm{Mpc}/h]^3$')
        plt.show()


        plt.figure()
        plt.semilogx(kk,Pk_neutrino/Pk_LambdaCDM-1)
        plt.xlabel(r'$k [h /\mathrm{Mpc}]$')
        plt.ylabel(r'$P(k)^\nu/P(k)-1$')
        plt.show()
    
        return neutrino_results,LambdaCDM_results
    else:
        return neutrino_results,LambdaCDM_results


# The function below takes 6 filenames as inputs in quotes. These are the names the files that will be saved. It takes three mass inputs. These are to be entered in meV. Lastly it requires a file as an input so it can load the e,f array as well as other important data from the file. 

# In[15]:


def make_data(filename1,filename2,filename3,filename4,filename5,filename6,small_m1,small_m2,small_m3,filename):
    m1 = small_m1*10**-3
    m2 = small_m2*10**-3
    m3 = small_m3*10**-3
    
    n_result_nontherm,L_result_nontherm = v_masses_nontherm_alpha(m1,True,filename,False,filename1,False)
    n_result2_nontherm,L_result2_nontherm = v_masses_nontherm_alpha(m2,True,filename,False,filename2,False)
    n_result3_nontherm,L_result3_nontherm = v_masses_nontherm_alpha(m3,True,filename,False,filename3,False)
    
    n_result3_therm,L_result3_therm = v_masses_therm(m1,True,filename4,False)
    n_result3_therm,L_result3_therm = v_masses_therm(m2,True,filename5,False)
    n_result3_therm,L_result3_therm = v_masses_therm(m3,True,filename6,False)


# This function takes the 6 filenames as inputs. Should be the 6 filenames entered in the previous function. Spits out the DeltaP/P plots. Graph should have 6 plots.

# In[10]:


def make_graphs(file1,file2,file3,file4,file5,file6):
    
    data_m1= np.load(file1)
    L_array_m1 = data_m1['LambdaCDM_array']
    Pkn_array_m1 = data_m1['Pk_n_array']
    k_array_m1 = data_m1['k_n_array']
    
    data_m2= np.load(file2)
    L_array_m2 = data_m2['LambdaCDM_array']
    Pkn_array_m2 = data_m2['Pk_n_array']
    k_array_m2 = data_m2['k_n_array']
    
    data_m3= np.load(file3)
    L_array_m3 = data_m3['LambdaCDM_array']
    Pkn_array_m3 = data_m3['Pk_n_array']
    k_array_m3 = data_m3['k_n_array']
    
    
    
    data_m4= np.load(file4)
    L_array_m4 = data_m4['Lambda_array']
    Pkn_array_m4 = data_m4['Pk_n_array']
    k_array_m4 = data_m4['k_n_array']
    
    data_m5= np.load(file5)
    L_array_m5 = data_m5['Lambda_array']
    Pkn_array_m5 = data_m5['Pk_n_array']
    k_array_m5 = data_m5['k_n_array']
    
    data_m6= np.load(file6)
    L_array_m6 = data_m6['Lambda_array']
    Pkn_array_m6 = data_m6['Pk_n_array']
    k_array_m6 = data_m6['k_n_array']
    
    

    plt.figure()
    plt.semilogx(k_array_m1,Pkn_array_m1/L_array_m1-1,color="red")
    plt.semilogx(k_array_m2,Pkn_array_m2/L_array_m2-1,color="purple")
    plt.semilogx(k_array_m3,Pkn_array_m3/L_array_m3-1,color="blue")
    plt.semilogx(k_array_m4,Pkn_array_m4/L_array_m4-1,linestyle='--',color="red")
    plt.semilogx(k_array_m5,Pkn_array_m5/L_array_m5-1,linestyle='--',color="purple")
    plt.semilogx(k_array_m6,Pkn_array_m6/L_array_m6-1,linestyle='--',color="blue")
    plt.ylabel(r'$P(k)^\nu/P(k)-1$')
    plt.show()
    


# In[ ]:




