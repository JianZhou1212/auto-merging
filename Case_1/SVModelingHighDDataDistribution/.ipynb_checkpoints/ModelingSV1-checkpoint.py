import numpy as np
import scipy.stats as stats

class ModelingSV1( ):
    def __init__(self, Params):
        self.L_Center  = Params['L_Center']
        self.w_lane    = Params['w_lane']
        self.N         = Params['N']
        self.DSV       = Params['DSV']
        self.T         = Params['T']
        self.max_speed = Params['max_speed']
        self.xi_HD     = Params['xi_HD']
        self.cdf_HD    = Params['cdf_HD']
        self.SV1_Min   = Params['SV1_Min']
        self.SV1_Max   = Params['SV1_Max']

    def Return(self, k, current_x_SV, current_x_SV0):
        L_Center = self.L_Center
        w_lane = self.w_lane
        N = self.N
        DSV = self.DSV
        T = self.T
        
        max_speed = self.max_speed - 10
        
        A_SV = np.array([[1, T], [0, 1]])
        B_SV = np.array([T**2/2, T]) 
        control_SV_horizon = self.inverse_transform_sampling(N)
        
        if (current_x_SV[1] >=  max_speed) or ((current_x_SV0[0] - current_x_SV[0])/current_x_SV[1] <= 1.2):
            control_SV_horizon = -np.abs(control_SV_horizon)
        x_SV_planning = np.zeros((DSV, N + 1))
        y_SV_planning = np.array([1.5*w_lane]*(N + 1))
        x_SV_planning[:, 0] = current_x_SV
        
        for t in range(1, N + 1):
            x_SV_planning[:, t] = A_SV@x_SV_planning[:, t-1] + B_SV*control_SV_horizon[t - 1]
                
        return control_SV_horizon, x_SV_planning, y_SV_planning
    
    def inverse_transform_sampling(self, N):
        xi_HD   = self.xi_HD
        cdf_HD  = self.cdf_HD
        SV1_Min = self.SV1_Min
        SV1_Max = self.SV1_Max

        uniform_random_values = np.random.uniform(SV1_Min, SV1_Max, N)
        samples = np.zeros(N)

        for i in range(N):
            index = (np.abs(cdf_HD - uniform_random_values[i])).argmin()
            samples[i] = xi_HD[index]

        return samples
        
        