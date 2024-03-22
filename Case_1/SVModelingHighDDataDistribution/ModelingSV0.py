#!/usr/bin/env python
# coding: utf-8

# In[4]:

import numpy as np
import scipy.stats as stats

class ModelingSV0( ):
    def __init__(self, Params):
        self.L_Center  = Params['L_Center']
        self.w_lane    = Params['w_lane']
        self.N         = Params['N']
        self.DSV       = Params['DSV']
        self.T         = Params['T']
        self.max_speed = Params['max_speed']
        self.xi_HD     = Params['xi_HD']
        self.cdf_HD    = Params['cdf_HD']
        self.SV0_Min   = Params['SV0_Min']
        self.SV0_Max   = Params['SV0_Max']
        

    def Return(self, k, current_x_SV):
        L_Center = self.L_Center
        w_lane = self.w_lane
        N = self.N
        DSV = self.DSV
        T = self.T
        max_speed = self.max_speed - 10
        
        A_SV = np.array([[1, T], [0, 1]])
        B_SV = np.array([T**2/2, T]) 

        control_SV_horizon = np.zeros(N)
        if 851 <= current_x_SV[0]:
            control_SV_horizon = self.inverse_transform_sampling(N)
            
        if current_x_SV[1] >=  max_speed:
            control_SV_horizon = control_SV_horizon*0    
            
        x_SV_planning = np.zeros((DSV, N + 1))
        y_SV_planning = np.array([1.5*w_lane]*(N + 1))
        x_SV_planning[:, 0] = current_x_SV
        
        for t in range(1, N + 1):
            x_SV_planning[:, t] = A_SV@x_SV_planning[:, t-1] + B_SV*control_SV_horizon[t - 1]
                
        return control_SV_horizon, x_SV_planning, y_SV_planning
    
    def inverse_transform_sampling(self, N):
        xi_HD   = self.xi_HD
        cdf_HD  = self.cdf_HD
        SV0_Min = self.SV0_Min
        SV0_Max = self.SV0_Max

        uniform_random_values = np.random.uniform(SV0_Min, SV0_Max, N)
        samples = np.zeros(N)

        for i in range(N):
            index = (np.abs(cdf_HD - uniform_random_values[i])).argmin()
            samples[i] = xi_HD[index]

        return samples
        
    
