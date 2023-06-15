import matplotlib.pyplot as plt
import numpy as np
from classy import Class

def v_masses(m_small, normal):
    delta_m21_sq = .0000750
    delta_m31_sq = 0.002458
    if normal:
        m1 = m_small
        m2_sq = delta_m21_sq + m1**2
        m3_sq = delta_m31_sq +m1**2
        m2 = np.sqrt(m2_sq)
        m3 = np.sqrt(m3_sq)
    else:
        m3 = m_small
        m1_sq = m3**2 + delta_m31_sq
        m2_sq = delta_m21_sq + m1_sq
        m1 = np.sqrt(m1_sq)
        m2 = np.sqrt(m2_sq)
        
    return m1,m2,m3

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
                     'background_verbose':1
                    }



LambdaCDM = Class()
LambdaCDM.set(LambdaCDM_settings)
LambdaCDM.compute()



kk = np.logspace(-4,np.log10(3),1000) # k in h/Mpc
Pk_LambdaCDM = np.zeros(len(kk)) # P(k) in (Mpc/h)**3
h = LambdaCDM_settings['h'] # get reduced Hubble for conversions to 1/Mpc

for i,k in enumerate(kk):
    Pk_LambdaCDM[i] = LambdaCDM.pk(k*h,0.)*h**3 # function .pk(k,z)
    
    
def dict_results(spectrum):
    Dict_S = {'age': spectrum.age(),
            'Neff': spectrum.Neff(),
            'omega_b': spectrum.omega_b(),
            'Omega0_cdm':spectrum.Omega0_cdm(),
            'h':spectrum.h(),
            'Omega0_k':spectrum.Omega0_k(),
            'Omega0_m': spectrum.Omega0_m(),
            'Omega_b': spectrum.Omega_b(),
            'Omega_g': spectrum.Omega_g(),
            'Omega_lambda': spectrum.Omega_Lambda(),
            'Omega_m': spectrum.Omega_m(),
            'Omega_r': spectrum.Omega_r(),
            'rs_drag': spectrum.rs_drag(),
            'Sigma8': spectrum.sigma8(),
            'Sigma8_cb': spectrum.sigma8_cb(),
            'T_cmb': spectrum.T_cmb(),
            'theta_s_100': spectrum.theta_s_100(),
            'theta_star_100': spectrum.theta_star_100(),  
            'n_s':spectrum.n_s(),
            'tau_reio':spectrum.tau_reio()
             }
    return Dict_S    

def v_masses_new(m_small, normal, filename):
    
    v_masses(m_small,normal)
    
    m1,m2,m3 = v_masses(m_small,normal)
    
    
    neutrino_mass_settings = {'N_ur':0.00441,
                              'N_ncdm':3,
                              'm_ncdm':'{},{},{}'.format(m1,m2,m3)     
                             }

    neutrino = Class()
    neutrino.set(LambdaCDM_settings)
    neutrino.set(neutrino_mass_settings)
    neutrino.compute()

    kk = np.logspace(-4,np.log10(3),1000) # k in h/Mpc
    Pk_neutrino = np.zeros(len(kk))

    for i,k in enumerate(kk):
        Pk_neutrino[i] = neutrino.pk(k*h,0.)*h**3 # function .pk(k,z)


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
    
    dict_n = dict_results(neutrino)
    print(dict_n)
    
    
    np.savez(filename, n_results = dict_n, k_n_array = kk, Pk_n_array = Pk_neutrino, truth_value = normal, v1 = m1, v2 = m2, v3 = m3, small_mass = m_small, sum_n = m1+m2+m3)
    