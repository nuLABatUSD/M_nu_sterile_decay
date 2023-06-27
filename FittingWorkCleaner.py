#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import numba as nb


# In[4]:


data_file = np.load("test-fit.npz", allow_pickle=True)
e = data_file['eps']
f = data_file['fe']


# In[5]:


data_file2 = np.load("test-fit2.npz", allow_pickle=True)
e2 = data_file2['eps']
f2 = data_file2['fe']


# In[6]:


def fit(e_array, f_array):
    e_max = e_array[np.where(e_array**2*f_array == np.max(e_array**2*f_array))[0]][0]
    f_max = f_array[np.where(e_array**2*f_array == np.max(e_array**2*f_array))[0]][0]
    
    T = e_max/2.301
    N = (((np.exp(e_max/T)+1)*np.max(e_array**2*f_array))/e_max**2)
    
    return T,N


# In[7]:


@nb.jit(nopython=True)
def least_sum(e_array,f_array,T, N):
    Sum = 0 
    for i in range(len(e_array)):
        Sum = Sum + ((e_array[i]**2*f_array[i]-((N)*(e_array[i]**2)/(np.exp(e_array[i]/T)+1)))**2)
    return Sum 
T = np.linspace(1.283,1.717,100)
N = np.linspace(-0.15,1.65,100)
M = np.zeros((len(T),len(N)))
for i in range(len(T)):
    for j in range(len(N)):
        M[i,j] = least_sum(e,f,T[i],N[j])


# In[8]:


@nb.jit(nopython=True)
def fit2(e_array, f_array):
    e_max = e_array[np.where(e_array**2*f_array == np.max(e_array**2*f_array))[0]][0]
    f_max = f_array[np.where(e_array**2*f_array == np.max(e_array**2*f_array))[0]][0]
    
    Del_e = e_array[1] - e_array[0]
    Del_T = (1/2.301)*Del_e
    
    T = e_max/2.301
    N = (((np.exp(e_max/T)+1)*np.max(e_array**2*f_array))/e_max**2)
    Del_N = (((np.exp(e_max/T+Del_T)+1)*np.max(e_array**2*f_array))/e_max**2)
    
    return T,N,Del_T,Del_N


# In[9]:


@nb.jit(nopython=True)
def fit3(e_array,f_array):
    
    fit2(e_array,f_array)
    
    T_0, N_0, T_error, N_error = fit2(e_array,f_array)
    
    
    T = np.linspace(T_0-T_error,T_0+T_error,100)
    N = np.linspace(N_0-N_error,N_0+N_error,100)
    M = np.zeros((len(T),len(N)))
    for i in range(len(T)):
        for j in range(len(N)):
            M[i,j] = least_sum(e_array,f_array,T[i],N[j])
    
    w = np.where(M==np.amin(M))
    T_best = T[w[0][0]]
    N_best = N[w[1][0]]
    
    return T_best,N_best


# In[22]:


def everything1(e_array,f_array):
    fit3(e_array,f_array)
    Tbest,Nbest = fit3(e_array,f_array)
    e_array_reverse = e_array[::-1]
    f_array_reverse = f_array[::-1]
    E_diff = np.zeros(len(e_array_reverse))
    
    for i in range(len(e_array_reverse)):
        E_diff[i] = (f_array_reverse[i])-((Nbest)/(np.exp(e_array_reverse[i]/Tbest)+1))
    return E_diff,Tbest,Nbest


# In[23]:


def everything2(e_array,f_array):
    
    everything1(e_array,f_array)
    
    E_smaller_reverse,T_best,N_best = everything1(e_array,f_array)
    
    E_smaller = []
    for i in E_smaller_reverse: 
        if i < 0:
            break
        if i > 0:
            E_smaller.append(i)
    E_smaller_diff = E_smaller[::-1]
    E_new = e_array[len(e_array)-len(E_smaller_diff):]
    F_new = f_array[len(e_array)-len(E_smaller_diff):]
    
    
    np.polyfit(E_new,np.log(E_smaller_diff),2) 
    A_best = np.polyfit(E_new,np.log(E_smaller_diff),2)[0]
    B_best = np.polyfit(E_new,np.log(E_smaller_diff),2)[1]
    C_best = np.polyfit(E_new,np.log(E_smaller_diff),2)[2]
    
    poly = A_best*e_array**2+B_best*e_array+C_best
    
    
    plt.figure()
    plt.semilogy(e_array,e_array**2*f_array,color="black")
    plt.semilogy(e_array,N_best*(e_array**2)/(np.exp(e_array/T_best)+1)+e**2*np.exp(poly),color="green")
    plt.show()
    
    return A_best,B_best,C_best
    


# In[24]:


everything2(e,f)


# In[13]:


everything2(e2,f2)


# In[ ]:




