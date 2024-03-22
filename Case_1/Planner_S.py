import numpy as np
import casadi
from pytope import Polytope

class Planner_S( ):
    def __init__(self, Params):
        
        self.T  = Params['T']
        self.N  = Params['N']
        self.Np = Params['Np']
        self.l_veh     = Params['l_veh']
        self.w_veh     = Params['w_veh']
        self.w_lane    = Params['w_lane']
        self.l_f       = Params['l_f']
        self.l_r       = Params['l_r']
        self.DEV       = Params['DEV']
        self.A_SV      = Params['A_SV']
        self.B_SV      = Params['B_SV']
        self.d_min_mpc = Params['d_min_mpc']
        self.A_road = Params['A_road']
        self.b_road = Params['b_road']
        self.infinity = Params['infinity']
        self.x_c = Params['x_c']
        self.y_c = Params['y_c']
        self.r_x = Params['r_x']
        self.r_y = Params['r_y']
        self.p   = Params['p']
        self.v_low     = Params['v_low']
        self.v_up      = Params['v_up']
        self.acc_low   = Params['acc_low']
        self.acc_up    = Params['acc_up']
        self.delta_low = Params['delta_low']
        self.delta_up  = Params['delta_up']
        self.Ref_Speed_PredictBySV = self.Ref_Speed_PredictBySV( )
        self.MPCFormulation        = self.MPCFormulation( )
    
    def SVPrediction(self, x_ini_sv, x_ini_ev): # velocity tracking model
        N      = self.N
        A_SV   = self.A_SV
        l_veh  = self.l_veh
        speed_tol = 55
        epsilon   = 0.001
        A_EV = A_SV

        x_ini_ev = np.array([x_ini_ev[0], x_ini_ev[3]*np.cos(x_ini_ev[2])])
        
        X_EV       = np.zeros((2, N + 1))
        X_EV[:, 0] = x_ini_ev
        
        for i in range(1, N + 1):
            X_EV[:, i] = A_EV@X_EV[:, i-1]
            
        if x_ini_ev[0] <= x_ini_sv[0]:
            X_SV       = np.zeros((2, N + 1))
            X_SV[:, 0] = x_ini_sv
            for i in range(1, N + 1):
                X_SV[:, i] = A_SV@X_SV[:, i-1]
            X_Opti_SV, X_Keep_SV, X_Slow_SV = X_SV, X_SV, X_SV
            v_ref_Opti_SV, v_ref_Keep_SV, v_ref_Slow_SV = x_ini_sv[1], x_ini_sv[1], x_ini_sv[1]
        else:
            v_ref_Opti_SV = self.Ref_Speed_PredictBySV(X_EV[0, 1::], x_ini_sv)
            v_ref_Opti_SV = v_ref_Opti_SV.full()
            v_ref_Opti_SV = v_ref_Opti_SV[0][0]
            
            if v_ref_Opti_SV < 0:
                v_ref_Opti_SV = 0
            if 55 < v_ref_Opti_SV:
                v_ref_Opti_SV = speed_tol
            
            v_ref_Keep_SV = x_ini_sv[1]
            v_ref_Slow_SV = np.minimum(0.5*v_ref_Opti_SV, 0.5*v_ref_Keep_SV)
                
            X_Opti_SV = self.SVVelocityTracking(v_ref_Opti_SV, x_ini_sv)
            X_Keep_SV = self.SVVelocityTracking(v_ref_Keep_SV, x_ini_sv)
            X_Slow_SV = self.SVVelocityTracking(v_ref_Slow_SV, x_ini_sv)
                                           
        Risk_Opti = np.sum((X_Opti_SV[0, 1::] - X_EV[0, 1::])**2 < l_veh**2)/N
        Risk_Keep = np.sum((X_Keep_SV[0, 1::] - X_EV[0, 1::])**2 < l_veh**2)/N
        Risk_Slow = np.sum((X_Slow_SV[0, 1::] - X_EV[0, 1::])**2 < l_veh**2)/N
        D_V_Opti = (v_ref_Opti_SV - x_ini_sv[1])**2/speed_tol**2
        D_V_Keep = (v_ref_Keep_SV - x_ini_sv[1])**2/speed_tol**2
        D_V_Slow = (v_ref_Slow_SV - x_ini_sv[1])**2/speed_tol**2
        
        J_Opti = 1/np.sqrt(D_V_Opti + Risk_Opti**2 + epsilon)
        J_Keep = 1/np.sqrt(D_V_Keep + Risk_Keep**2 + epsilon)
        J_Slow = 1/np.sqrt(D_V_Slow + Risk_Slow**2 + epsilon)
        
        P_Opti = J_Opti/(J_Opti + J_Keep + J_Slow)
        P_Keep = J_Keep/(J_Opti + J_Keep + J_Slow)
        P_Slow = J_Slow/(J_Opti + J_Keep + J_Slow)
            
        return X_Opti_SV, X_Keep_SV, X_Slow_SV,  v_ref_Opti_SV, v_ref_Keep_SV, v_ref_Slow_SV, P_Opti, P_Keep, P_Slow
    
    def SVVelocityTracking(self, v_ref, x_ini_sv):
        N = self.N
        A_SV = self.A_SV
        B_SV = self.B_SV
        K_SV = 1

        X_SV  = np.zeros((2, N + 1))
        X_SV[:, 0] = x_ini_sv

        for i in range(1, N + 1):
            X_SV[:, i] = A_SV@X_SV[:, i-1] + B_SV*(-K_SV*(X_SV[1, i-1] - v_ref))
            
        return X_SV
        
    
    def Ref_Speed_PredictBySV(self):
        N = self.N
        A_SV = self.A_SV
        B_SV = self.B_SV
        l_veh = self.l_veh
        K_SV = 1

        opti = casadi.Opti( )
        x_ev      = opti.parameter(1, N)
        x_ini_sv  = opti.parameter(2, 1)
        v_current = x_ini_sv[1]
    
        v_ref = opti.variable( )
        x_sv  = opti.variable(2, N + 1)
        opti.subject_to(x_sv[:, 0] == x_ini_sv)

        for i in range(1, N + 1):
            x_sv[:, i] = A_SV@x_sv[:, i-1] + B_SV@(-K_SV*(x_sv[1, i-1] - v_ref))
            
        J = (v_current - v_ref)**2
        opti.minimize(J)
        for i in range(N):
            opti.subject_to(l_veh**2 <= (x_ev[i] - x_sv[0, i + 1])**2)
            
        opts = {"ipopt.print_level": 0, "ipopt.linear_solver": "ma57", "print_time": False}
        opti.solver('ipopt', opts)
        
        return opti.to_function('f', [x_ev, x_ini_sv], [v_ref])
        
    def SVOccupancy(self, X_SV):
        T = self.T
        N = self.N
        l_veh = self.l_veh
        w_veh = self.w_veh
        w_lane = self.w_lane

        G = np.zeros((4, 2*N))
        g = np.zeros((4, N))

        for t in range(1, N + 1):
            x = X_SV[:, t]
            min_x = x[0] - l_veh
            max_x = x[0] + l_veh
            min_y = 1.5*w_lane - w_veh
            max_y = 2*w_lane

            temp_poly = Polytope(np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]))
            G[:, 2*t-2:2*t] = temp_poly.A
            g[:, t-1]       = temp_poly.b.reshape(4, )
        
        return G, g
    
    def Return(self, current_x_SV, current_x_EV):
        
        w_lane = self.w_lane

        X_Opti_0, X_Keep_0, X_Slow_0, v_ref_Opti_SV_0, v_ref_Keep_SV_0, v_ref_Slow_SV_0, P_Opti_0, P_Keep_0, P_Slow_0 = self.SVPrediction(current_x_SV[0], current_x_EV)
        G_0_o, g_0_o = self.SVOccupancy(X_Opti_0)
        G_0_k, g_0_k = self.SVOccupancy(X_Keep_0)
        G_0_s, g_0_s = self.SVOccupancy(X_Slow_0)
        
        X_Opti_1, X_Keep_1, X_Slow_1, v_ref_Opti_SV_1, v_ref_Keep_SV_1, v_ref_Slow_SV_1, P_Opti_1, P_Keep_1, P_Slow_1 = self.SVPrediction(current_x_SV[1], current_x_EV)
        G_1_o, g_1_o = self.SVOccupancy(X_Opti_1)
        G_1_k, g_1_k = self.SVOccupancy(X_Keep_1)
        G_1_s, g_1_s = self.SVOccupancy(X_Slow_1)
                
        D_EV_SV0 = np.sqrt((current_x_EV[0] - current_x_SV[0][0])**2 + (current_x_EV[1] - 1.5*w_lane)**2)
        D_EV_SV1 = np.sqrt((current_x_EV[0] - current_x_SV[1][0])**2 + (current_x_EV[1] - 1.5*w_lane)**2)
        Risk_0 = D_EV_SV1/(D_EV_SV0 + D_EV_SV1)
        Risk_1 = D_EV_SV0/(D_EV_SV0 + D_EV_SV1)
        v_nom  = current_x_SV[0][1]
        X_0_o, X_0_k, X_0_s, X_1_o, X_1_k, X_1_s, U_0_o, U_0_k, U_0_s, U_1_o, U_1_k, U_1_s = self.MPCFormulation(v_nom, Risk_0, Risk_1, G_0_o, g_0_o, G_0_k, g_0_k, G_0_s, g_0_s, G_1_o, g_1_o, G_1_k, g_1_k, G_1_s, g_1_s, P_Opti_0, P_Keep_0, P_Slow_0, P_Opti_1, P_Keep_1, P_Slow_1, current_x_EV)
        X_0_o = X_0_o.full( )
        U_0_o = U_0_o.full( )
        X_0_k = X_0_k.full( )
        U_0_k = U_0_k.full( )
        X_0_s = X_0_s.full( )
        U_0_s = U_0_s.full( )
        X_1_o = X_1_o.full( )
        U_1_o = U_1_o.full( )
        X_1_k = X_1_k.full( )
        U_1_k = U_1_k.full( )
        X_1_s = X_1_s.full( )
        U_1_s = U_1_s.full( )
        
        return X_0_o, X_0_k, X_0_s, X_1_o, X_1_k, X_1_s, U_0_o[:, 0], v_ref_Opti_SV_0, v_ref_Keep_SV_0, v_ref_Slow_SV_0, P_Opti_0, P_Keep_0, P_Slow_0, v_ref_Opti_SV_1, v_ref_Keep_SV_1, v_ref_Slow_SV_1, P_Opti_1, P_Keep_1, P_Slow_1
    
    def MPCFormulation(self):
        N      = self.N
        Np     = self.Np
        DEV    = self.DEV
        d_min  = self.d_min_mpc
        T      = self.T
        w_lane = self.w_lane
        Q1 = 10
        Q2 = 10
        Q3 = 0.5
        Q4 = 0.1
        A_road    = self.A_road
        b_road    = self.b_road
        v_low     = self.v_low 
        v_up      = self.v_up 
        acc_low   = self.acc_low 
        acc_up    = self.acc_up 
        delta_low = self.delta_low 
        delta_up  = self.delta_up
        x_c = self.x_c
        y_c = self.y_c
        r_x = self.r_x
        r_y = self.r_y
        p   = self.p 

        opti  = casadi.Opti( )
        X_0_o = opti.variable(DEV, Np + 1)
        U_0_o = opti.variable(2, Np)
        X_0_k = opti.variable(DEV, Np + 1)
        U_0_k = opti.variable(2, Np)
        X_0_s = opti.variable(DEV, Np + 1)
        U_0_s = opti.variable(2, Np)
        
        X_1_o = opti.variable(DEV, Np + 1)
        U_1_o = opti.variable(2, Np)
        X_1_k = opti.variable(DEV, Np + 1)
        U_1_k = opti.variable(2, Np)
        X_1_s = opti.variable(DEV, Np + 1)
        U_1_s = opti.variable(2, Np)
        
        delta_0_o = U_0_o[0, :]
        eta_0_o   = U_0_o[1, :]
        delta_0_k = U_0_k[0, :]
        eta_0_k   = U_0_k[1, :]
        delta_0_s = U_0_s[0, :]
        eta_0_s   = U_0_s[1, :]
        
        delta_1_o = U_1_o[0, :]
        eta_1_o   = U_1_o[1, :]
        delta_1_k = U_1_k[0, :]
        eta_1_k   = U_1_k[1, :]
        delta_1_s = U_1_s[0, :]
        eta_1_s   = U_1_s[1, :]
        
        lam_0_o = opti.variable(4, Np)
        lam_0_k = opti.variable(4, Np)
        lam_0_s = opti.variable(4, Np)
        lam_1_o = opti.variable(4, Np)
        lam_1_k = opti.variable(4, Np)
        lam_1_s = opti.variable(4, Np)
        
        G_0_o = opti.parameter(4, 2*N)
        g_0_o = opti.parameter(4, N)
        G_0_k = opti.parameter(4, 2*N)
        g_0_k = opti.parameter(4, N)
        G_0_s = opti.parameter(4, 2*N)
        g_0_s = opti.parameter(4, N)
        G_1_o = opti.parameter(4, 2*N)
        g_1_o = opti.parameter(4, N)
        G_1_k = opti.parameter(4, 2*N)
        g_1_k = opti.parameter(4, N)
        G_1_s = opti.parameter(4, 2*N)
        g_1_s = opti.parameter(4, N)
        
        P_Opti_0 = opti.parameter( )
        P_Keep_0 = opti.parameter( )
        P_Slow_0 = opti.parameter( )
        P_Opti_1 = opti.parameter( )
        P_Keep_1 = opti.parameter( )
        P_Slow_1 = opti.parameter( )
        
        Risk_0 = opti.parameter( )
        Risk_1 = opti.parameter( )
        v_nom  = opti.parameter( )
        
        Initial = opti.parameter(DEV, 1)
        
        opti.subject_to(X_0_o[:, 0] == Initial)
        opti.subject_to(X_0_k[:, 0] == Initial)
        opti.subject_to(X_0_s[:, 0] == Initial)
        opti.subject_to(X_1_o[:, 0] == Initial)
        opti.subject_to(X_1_k[:, 0] == Initial)
        opti.subject_to(X_1_s[:, 0] == Initial)
        for k in range(Np):
            k1 = self.vehicle_model(X_0_o[:, k],          delta_0_o[k], eta_0_o[k])
            k2 = self.vehicle_model(X_0_o[:, k] + T/2*k1, delta_0_o[k], eta_0_o[k])
            k3 = self.vehicle_model(X_0_o[:, k] + T/2*k2, delta_0_o[k], eta_0_o[k])
            k4 = self.vehicle_model(X_0_o[:, k] + T*k3,   delta_0_o[k], eta_0_o[k])
            x_next = X_0_o[:, k] + T/6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X_0_o[:, k + 1] == x_next) 
            
            k1 = self.vehicle_model(X_0_k[:, k],          delta_0_k[k], eta_0_k[k])
            k2 = self.vehicle_model(X_0_k[:, k] + T/2*k1, delta_0_k[k], eta_0_k[k])
            k3 = self.vehicle_model(X_0_k[:, k] + T/2*k2, delta_0_k[k], eta_0_k[k])
            k4 = self.vehicle_model(X_0_k[:, k] + T*k3,   delta_0_k[k], eta_0_k[k])
            x_next = X_0_k[:, k] + T/6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X_0_k[:, k + 1] == x_next) 
            
            k1 = self.vehicle_model(X_0_s[:, k],          delta_0_s[k], eta_0_s[k])
            k2 = self.vehicle_model(X_0_s[:, k] + T/2*k1, delta_0_s[k], eta_0_s[k])
            k3 = self.vehicle_model(X_0_s[:, k] + T/2*k2, delta_0_s[k], eta_0_s[k])
            k4 = self.vehicle_model(X_0_s[:, k] + T*k3,   delta_0_s[k], eta_0_s[k])
            x_next = X_0_s[:, k] + T/6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X_0_s[:, k + 1] == x_next) 
            
            k1 = self.vehicle_model(X_1_o[:, k],          delta_1_o[k], eta_1_o[k])
            k2 = self.vehicle_model(X_1_o[:, k] + T/2*k1, delta_1_o[k], eta_1_o[k])
            k3 = self.vehicle_model(X_1_o[:, k] + T/2*k2, delta_1_o[k], eta_1_o[k])
            k4 = self.vehicle_model(X_1_o[:, k] + T*k3,   delta_1_o[k], eta_1_o[k])
            x_next = X_1_o[:, k] + T/6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X_1_o[:, k + 1] == x_next) 
            
            k1 = self.vehicle_model(X_1_k[:, k],          delta_1_k[k], eta_1_k[k])
            k2 = self.vehicle_model(X_1_k[:, k] + T/2*k1, delta_1_k[k], eta_1_k[k])
            k3 = self.vehicle_model(X_1_k[:, k] + T/2*k2, delta_1_k[k], eta_1_k[k])
            k4 = self.vehicle_model(X_1_k[:, k] + T*k3,   delta_1_k[k], eta_1_k[k])
            x_next = X_1_k[:, k] + T/6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X_1_k[:, k + 1] == x_next) 
            
            k1 = self.vehicle_model(X_1_s[:, k],          delta_1_s[k], eta_1_s[k])
            k2 = self.vehicle_model(X_1_s[:, k] + T/2*k1, delta_1_s[k], eta_1_s[k])
            k3 = self.vehicle_model(X_1_s[:, k] + T/2*k2, delta_1_s[k], eta_1_s[k])
            k4 = self.vehicle_model(X_1_s[:, k] + T*k3,   delta_1_s[k], eta_1_s[k])
            x_next = X_1_s[:, k] + T/6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X_1_s[:, k + 1] == x_next) 
            
            
        # collision-avoidance
        for k in range(Np):
            p_point_0_o = X_0_o[0:2, k + 1]
            p_point_0_k = X_0_k[0:2, k + 1]
            p_point_0_s = X_0_s[0:2, k + 1]
            p_point_1_o = X_1_o[0:2, k + 1]
            p_point_1_k = X_1_k[0:2, k + 1]
            p_point_1_s = X_1_s[0:2, k + 1]
            
            G_0_point_o = G_0_o[:, 2*k:2*k + 2]
            g_0_point_o = g_0_o[:, k]
            temp_0_o    = G_0_point_o.T@lam_0_o[:, k]
            
            G_0_point_k = G_0_k[:, 2*k:2*k + 2]
            g_0_point_k = g_0_k[:, k]
            temp_0_k    = G_0_point_k.T@lam_0_k[:, k]
            
            G_0_point_s = G_0_s[:, 2*k:2*k + 2]
            g_0_point_s = g_0_s[:, k]
            temp_0_s    = G_0_point_s.T@lam_0_s[:, k]
            
            G_1_point_o = G_1_o[:, 2*k:2*k + 2]
            g_1_point_o = g_1_o[:, k]
            temp_1_o    = G_1_point_o.T@lam_1_o[:, k]
            
            G_1_point_k = G_1_k[:, 2*k:2*k + 2]
            g_1_point_k = g_1_k[:, k]
            temp_1_k    = G_1_point_k.T@lam_1_k[:, k]
            
            G_1_point_s = G_1_s[:, 2*k:2*k + 2]
            g_1_point_s = g_1_s[:, k]
            temp_1_s    = G_1_point_s.T@lam_1_s[:, k]
            
            opti.subject_to((G_0_point_o@p_point_0_o - g_0_point_o).T@lam_0_o[:, k] >= d_min)
            opti.subject_to((temp_0_o[0]**2 + temp_0_o[1]**2) <= 1)
            opti.subject_to(0 <= lam_0_o[:, k])

            opti.subject_to((G_0_point_k@p_point_0_k - g_0_point_k).T@lam_0_k[:, k] >= d_min)
            opti.subject_to((temp_0_k[0]**2 + temp_0_k[1]**2) <= 1)
            opti.subject_to(0 <= lam_0_k[:, k])
            
            opti.subject_to((G_0_point_s@p_point_0_s - g_0_point_s).T@lam_0_s[:, k] >= d_min)
            opti.subject_to((temp_0_s[0]**2 + temp_0_s[1]**2) <= 1)
            opti.subject_to(0 <= lam_0_s[:, k])
            
            opti.subject_to((G_1_point_o@p_point_1_o - g_1_point_o).T@lam_1_o[:, k] >= d_min)
            opti.subject_to((temp_1_o[0]**2 + temp_1_o[1]**2) <= 1)
            opti.subject_to(0 <= lam_1_o[:, k])
            
            opti.subject_to((G_1_point_k@p_point_1_k - g_1_point_k).T@lam_1_k[:, k] >= d_min)
            opti.subject_to((temp_1_k[0]**2 + temp_1_k[1]**2) <= 1)
            opti.subject_to(0 <= lam_1_k[:, k])
            
            opti.subject_to((G_1_point_s@p_point_1_s - g_1_point_s).T@lam_1_s[:, k] >= d_min)
            opti.subject_to((temp_1_s[0]**2 + temp_1_s[1]**2) <= 1)
            opti.subject_to(0 <= lam_1_s[:, k])
            
            opti.subject_to(1.1 <= ((p_point_0_o[0] - x_c)/r_x)**p + ((p_point_0_o[1] - y_c)/r_y)**p)
            opti.subject_to(1.1 <= ((p_point_0_k[0] - x_c)/r_x)**p + ((p_point_0_k[1] - y_c)/r_y)**p)
            opti.subject_to(1.1 <= ((p_point_0_s[0] - x_c)/r_x)**p + ((p_point_0_s[1] - y_c)/r_y)**p)
            
            opti.subject_to(1.1 <= ((p_point_1_o[0] - x_c)/r_x)**p + ((p_point_1_o[1] - y_c)/r_y)**p)
            opti.subject_to(1.1 <= ((p_point_1_k[0] - x_c)/r_x)**p + ((p_point_1_k[1] - y_c)/r_y)**p)
            opti.subject_to(1.1 <= ((p_point_1_s[0] - x_c)/r_x)**p + ((p_point_1_s[1] - y_c)/r_y)**p)
            
            opti.subject_to(A_road@p_point_0_o <= b_road)
            opti.subject_to(A_road@p_point_0_k <= b_road)
            opti.subject_to(A_road@p_point_0_s <= b_road)
            
            opti.subject_to(A_road@p_point_1_o <= b_road)
            opti.subject_to(A_road@p_point_1_k <= b_road)
            opti.subject_to(A_road@p_point_1_s <= b_road)
        
        y_0_o = X_0_o[1, 1::]
        v_0_o = X_0_o[3, 1::]
        a_0_o = X_0_o[4, 1::]
        y_0_k = X_0_k[1, 1::]
        v_0_k = X_0_k[3, 1::]
        a_0_k = X_0_k[4, 1::]
        y_0_s = X_0_s[1, 1::]
        v_0_s = X_0_s[3, 1::]
        a_0_s = X_0_s[4, 1::]
        
        y_1_o = X_1_o[1, 1::]
        v_1_o = X_1_o[3, 1::]
        a_1_o = X_1_o[4, 1::]
        y_1_k = X_1_k[1, 1::]
        v_1_k = X_1_k[3, 1::]
        a_1_k = X_1_k[4, 1::]
        y_1_s = X_1_s[1, 1::]
        v_1_s = X_1_s[3, 1::]
        a_1_s = X_1_s[4, 1::]
        
        v_0_o_e = v_0_o[-1] - v_nom
        v_0_k_e = v_0_k[-1] - v_nom
        v_0_s_e = v_0_s[-1] - v_nom
        v_1_o_e = v_1_o[-1] - v_nom
        v_1_k_e = v_1_k[-1] - v_nom
        v_1_s_e = v_1_s[-1] - v_nom
        
        y_0_o_e = y_0_o[-1] - 1.5*w_lane
        y_0_k_e = y_0_k[-1] - 1.5*w_lane
        y_0_s_e = y_0_s[-1] - 1.5*w_lane
        y_1_o_e = y_1_o[-1] - 1.5*w_lane
        y_1_k_e = y_1_k[-1] - 1.5*w_lane
        y_1_s_e = y_1_s[-1] - 1.5*w_lane
        
        opti.subject_to(U_0_o[:, 0] == U_0_k[:, 0])
        opti.subject_to(U_0_k[:, 0] == U_0_s[:, 0])
        
        opti.subject_to(U_1_o[:, 0] == U_1_k[:, 0])
        opti.subject_to(U_1_k[:, 0] == U_1_s[:, 0])
        
        opti.subject_to(U_0_o[:, 0] == U_1_o[:, 0])
        
        opti.subject_to(opti.bounded(v_low, v_0_o, v_up))
        opti.subject_to(opti.bounded(acc_low, a_0_o, acc_up))
        opti.subject_to(opti.bounded(delta_low, delta_0_o, delta_up))
        opti.subject_to(opti.bounded(v_low, v_0_k, v_up))
        opti.subject_to(opti.bounded(acc_low, a_0_k, acc_up))
        opti.subject_to(opti.bounded(delta_low, delta_0_k, delta_up))
        opti.subject_to(opti.bounded(v_low, v_0_s, v_up))
        opti.subject_to(opti.bounded(acc_low, a_0_s, acc_up))
        opti.subject_to(opti.bounded(delta_low, delta_0_s, delta_up))
        
        opti.subject_to(opti.bounded(v_low, v_1_o, v_up))
        opti.subject_to(opti.bounded(acc_low, a_1_o, acc_up))
        opti.subject_to(opti.bounded(delta_low, delta_1_o, delta_up))
        opti.subject_to(opti.bounded(v_low, v_1_k, v_up))
        opti.subject_to(opti.bounded(acc_low, a_1_k, acc_up))
        opti.subject_to(opti.bounded(delta_low, delta_1_k, delta_up))
        opti.subject_to(opti.bounded(v_low, v_1_s, v_up))
        opti.subject_to(opti.bounded(acc_low, a_1_s, acc_up))
        opti.subject_to(opti.bounded(delta_low, delta_1_s, delta_up))
        
        J_0_o = P_Opti_0*(delta_0_o@Q1@delta_0_o.T + eta_0_o@Q2@eta_0_o.T + v_0_o_e@Q3@v_0_o_e.T + y_0_o_e@Q4@y_0_o_e.T)  
        J_0_k = P_Keep_0*(delta_0_k@Q1@delta_0_k.T + eta_0_k@Q2@eta_0_k.T + v_0_k_e@Q3@v_0_k_e.T + y_0_k_e@Q4@y_0_k_e.T) 
        J_0_s = P_Slow_0*(delta_0_s@Q1@delta_0_s.T + eta_0_s@Q2@eta_0_s.T + v_0_s_e@Q3@v_0_s_e.T + y_0_s_e@Q4@y_0_s_e.T)
        J_1_o = P_Opti_1*(delta_1_o@Q1@delta_1_o.T + eta_1_o@Q2@eta_1_o.T + v_1_o_e@Q3@v_1_o_e.T + y_1_o_e@Q4@y_1_o_e.T) 
        J_1_k = P_Keep_1*(delta_1_k@Q1@delta_1_k.T + eta_1_k@Q2@eta_1_k.T + v_1_k_e@Q3@v_1_k_e.T + y_1_k_e@Q4@y_1_k_e.T) 
        J_1_s = P_Slow_1*(delta_1_s@Q1@delta_1_s.T + eta_1_s@Q2@eta_1_s.T + v_1_s_e@Q3@v_1_s_e.T + y_1_s_e@Q4@y_1_s_e.T)
        J = Risk_0*(J_0_o + J_0_k + J_0_s) + Risk_1*(J_1_o + J_1_k + J_1_s)
        opti.minimize(J)
        
        opts = {"ipopt.print_level": 0,"ipopt.linear_solver": "ma57","print_time": False}
        opti.solver('ipopt', opts)

        return opti.to_function('g', [v_nom, Risk_0, Risk_1, G_0_o, g_0_o, G_0_k, g_0_k, G_0_s, g_0_s, G_1_o, g_1_o, G_1_k, g_1_k, G_1_s, g_1_s, P_Opti_0, P_Keep_0, P_Slow_0, P_Opti_1, P_Keep_1, P_Slow_1, Initial], [X_0_o, X_0_k, X_0_s, X_1_o, X_1_k, X_1_s, U_0_o, U_0_k, U_0_s, U_1_o, U_1_k, U_1_s])

    def  vehicle_model(self, w, delta, eta):
        l_f = self.l_f
        l_r = self.l_r
        
        x_dot = w[3] 
        y_dot = w[3]*w[2] + l_r/(l_f + l_r)*w[3]*delta
        phi_dot = w[3]/(l_f + l_r)*delta
        v_dot = w[4]
        a_dot = eta
        
        return casadi.vertcat(x_dot, y_dot, phi_dot, v_dot, a_dot)