import numpy as np
import casadi
from pytope import Polytope
from scipy import interpolate
from numpy.linalg import matrix_power


class Planner_P( ):
    def __init__(self, Params):
        
        self.T = Params['T']
        self.N = Params['N']
        self.Np = Params['Np']
        self.N_coarse = Params['N_coarse']
        self.T_coarse = Params['T_coarse']
        self.l_veh = Params['l_veh']
        self.w_veh = Params['w_veh']
        self.l_f = Params['l_f']
        self.l_r = Params['l_r']
        self.w_lane = Params['w_lane']
        self.DEV = Params['DEV']
        self.DPM = Params['DPM']
        self.N_Lane = Params['N_Lane']
        self.N_M = Params['N_M']
        self.N_Car = Params['N_Car']
        self.L_Bound = Params['L_Bound']
        self.L_Center = Params['L_Center']
        self.K_Lon_EV = Params['K_Lon_EV']
        self.K_Lat_EV = Params['K_Lat_EV']
        self.SpeedNom = Params['SpeedNom']
        self.Weight = Params['Weight']
        self.A_SV_coarse = Params['A_SV_coarse']
        self.B_SV_coarse = Params['B_SV_coarse']
        self.X_SV_Poly = Params['X_SV_Poly']
        self.infinity = Params['infinity']
        self.max_speed = Params['max_speed']
        self.road_terminal = Params['road_terminal']
        self.terminal_margin = Params['terminal_margin']
        self.d_min = Params['d_min']
        self.d_min_mpc = Params['d_min_mpc']
        self.Q1 = Params['Q1']
        self.Q2 = Params['Q2']
        self.Q3 = Params['Q3']
        self.Q4 = Params['Q4']
        self.v_low = Params['v_low']
        self.v_up = Params['v_up']
        self.acc_low = Params['acc_low']
        self.acc_up = Params['acc_up']
        self.delta_low = Params['delta_low']
        self.delta_up = Params['delta_up']
        self.Ref_Speed_Stage    = self.Ref_Speed_Stage( )
        self.MPCFormulation = self.MPCFormulation( )
    
    def VelocityTracking(self, x_ini, vx_ref, m, n_step): # velocity tracking model
        T = self.T
        L_Center = self.L_Center
        DPM = self.DPM
        K_Lon = self.K_Lon_EV
        K_Lat = self.K_Lat_EV
        k_lo_1 = K_Lon[0]
        k_lo_2 = K_Lon[1]
        k_la_1 = K_Lat[0]
        k_la_2 = K_Lat[1]
        k_la_3 = K_Lat[2] 
        y_ref = L_Center[m]
        F = np.array([[1, T, T**2/2, 0, 0, 0],
                      [0, 1-k_lo_1*T**2/2, T-k_lo_2*(T**2)/2, 0, 0, 0],
                      [0, -k_lo_1*T, 1-k_lo_2*T, 0, 0, 0],
                      [0, 0, 0, 1-k_la_1*(T**3)/6, T-k_la_2*(T**3)/6, T**2/2-k_la_3*(T**3)/6],
                      [0, 0, 0, -k_la_1*(T**2)/2, 1-k_la_2*(T**2)/2, T-k_la_3*(T**2)/2],
                      [0, 0, 0, -k_la_1*T, -k_la_2*T, 1-k_la_3*T]]) # shape = 6 x 6
        E = np.array([0, k_lo_1*(T**2)/2*vx_ref, k_lo_1*T*vx_ref, (T**3)/6*k_la_1*y_ref, (T**2)/2*k_la_1*y_ref, T*k_la_1*y_ref]) # array no shape
        X_KF = np.zeros((DPM, n_step+1))
        X_KF[:, 0] = x_ini
            
        for i in range(1, n_step+1):
            X_KF[:, i] = (F@X_KF[:, i-1]) + E
            
        return X_KF
        
    def LaneTracking(self, initial_y, m): # lane tracking model
        L_Center = self.L_Center
        T = self.T
        N = self.N
        K_Lat = self.K_Lat_EV
        k_la_1 = K_Lat[0]
        k_la_2 = K_Lat[1]
        k_la_3 = K_Lat[2]
        y_ref = L_Center[m]
        A = np.array([[1-k_la_1*(T**3)/6, T-k_la_2*(T**3)/6, (T**2)/2-k_la_3*(T**3)/6],
                      [-k_la_1*(T**2)/2, 1-k_la_2*(T**2)/2, T-k_la_3*(T**2)/2],
                      [-k_la_1*T, -k_la_2*T, 1-k_la_3*T]])
        B = np.array([(T**3)/6*k_la_1, (T**2)/2*k_la_1, T*k_la_1])*y_ref
        Y = np.zeros((3, N + 1))
        Y[:, 0] = initial_y
        
        for i in range(1, N+1):
            Y[:, i] =  (A@Y[:, i-1]) + B    
        
        return Y[0, :] # returns back longitudinal position, a row
    
    def ProjectSpeed(self, Robust_SV_Position_k, x_SV_k, x_hat_k, RefPrim):
        road_terminal = self.road_terminal
        terminal_margin = self.terminal_margin
        l_veh = self.l_veh
        infinity = self.infinity
        N_M = self.N_M
        N = self.N
        T = self.T
        K_Lon = self.K_Lon_EV
        d_min = self.d_min
        
        a = np.array([[1, T, (T**2)/2], [0, 1-K_Lon[0]*(T**2)/2, T-K_Lon[1]*(T**2)/2], [0, -K_Lon[0]*T, 1-K_Lon[1]*T]])
        b = np.array([0, K_Lon[0]*(T**2)/2, K_Lon[0]*T])
        EV_x_pos = x_hat_k[0]
        EV_x_vel = x_hat_k[1]
        
        SV0_pos = x_SV_k[0][0]
        SV1_pos = x_SV_k[1][0]
        
        SafeVal = list( ) # save the safe speed
        SafeVal_All = np.array([None]*3)
        initial_x = x_hat_k[0:3] # x part
        A_x = list( )
        B_x = list( )
        A_v = list( )
        B_v = list( )
        sel_x = np.array([[1, 0, 0]]) # select long. position
        sel_v = np.array([[0, 1, 0]]) # select long. velocity
        A_x.append(sel_x@b)
        B_x.append(sel_x@matrix_power(a, 1)@initial_x)
        A_v.append(sel_v@b)
        B_v.append(sel_v@matrix_power(a, 1)@initial_x)
        for j in range(1, N):
            A_x.append(sel_x@matrix_power(a, j)@b + A_x[j - 1])
            B_x.append(sel_x@matrix_power(a, j + 1)@initial_x)
            A_v.append(sel_v@matrix_power(a, j)@b + A_v[j - 1])
            B_v.append(sel_v@matrix_power(a, j + 1)@initial_x)
        A_x = np.array(A_x).reshape(N, 1)
        B_x = np.array(B_x).reshape(N, 1)
        A_v = np.array(A_v).reshape(N, 1)
        B_v = np.array(B_v).reshape(N, 1)
        
        for i in range(N_M):
            if i == 0: # VT1
                if EV_x_pos + l_veh/2 < road_terminal[0]:
                    X_lim_low = np.ones((N, 1))*(-infinity)
                    X_lim_up = np.ones((N, 1))*road_terminal[0] - terminal_margin
                    vx_up = self.Ref_Speed_Stage(X_lim_low, X_lim_up, A_v, B_v, A_x, B_x, EV_x_vel)
                    vx_up = vx_up.__float__( )
                    if vx_up < 0:
                        vx_up = 0
                else:
                    vx_up = -10
                SafeVal.append(vx_up)
            else: # VT2
                if SV0_pos <= EV_x_pos:
                    X_lim_low = Robust_SV_Position_k[0][0, :] + Robust_SV_Position_k[0][2, :]
                    X_lim_up = np.ones((N, 1))*infinity
                    vx_up= self.Ref_Speed_Stage(X_lim_low.reshape(N, 1), X_lim_up, A_v, B_v, A_x, B_x, EV_x_vel)
                    vx_up= vx_up.__float__( )
                    if vx_up < 0:
                        vx_up = 0
                    SafeVal_All[0] = vx_up
                elif EV_x_pos <= SV1_pos:
                    X_lim_up = Robust_SV_Position_k[1][0, :] - Robust_SV_Position_k[1][2, :]
                    X_lim_low = np.ones((N, 1))*(-infinity)
                    vx_up= self.Ref_Speed_Stage(X_lim_low, X_lim_up.reshape(N, 1), A_v, B_v, A_x, B_x, EV_x_vel)
                    vx_up= vx_up.__float__( )
                    if vx_up < 0:
                        vx_up = 0
                    SafeVal_All[1] = vx_up
                else:
                    X_lim_up = Robust_SV_Position_k[0][0, :] - Robust_SV_Position_k[0][2, :]
                    X_lim_low = Robust_SV_Position_k[1][0, :] + Robust_SV_Position_k[1][2, :]
                    if np.min(X_lim_up - X_lim_low) <= 2*d_min:
                        vx_up = 0
                    else:
                        vx_up= self.Ref_Speed_Stage(X_lim_low.reshape(N, 1), X_lim_up.reshape(N, 1), A_v, B_v, A_x, B_x, EV_x_vel)
                        vx_up= vx_up.__float__( )
                        if vx_up < 0:
                            vx_up = 0
                    SafeVal_All[2] = vx_up
                    
                SafeVal.append(vx_up)
                            
        SafeVal = np.array(SafeVal)

        return SafeVal
    
    def ReachableSet(self, current_x_SV, current_y_SV, samples):
        T = self.T
        N = self.N
        N_Car    = self.N_Car
        N_coarse = self.N_coarse
        T_coarse = self.T_coarse
        l_veh  = self.l_veh
        w_veh  = self.w_veh
        w_lane = self.w_lane
        A_SV_coarse  = self.A_SV_coarse
        B_SV_coarse  = self.B_SV_coarse
        X_SV_Poly    = self.X_SV_Poly
       
        Reachable_Set_Reduced = list( )
        Robust_SV_Position    = list( )
        G = list( )
        g = list( )
        
        for i in range(N_Car - 1):
            low_ax_reduced = np.min(samples[i])
            up_ax_reduced  = np.max(samples[i])
            BU_SV_Reduced_Poly = Polytope(np.array([[B_SV_coarse[0]*low_ax_reduced, B_SV_coarse[1]*low_ax_reduced], [B_SV_coarse[0]*up_ax_reduced, B_SV_coarse[1]*up_ax_reduced]])) # line segment in R^2
            coarse_length_min = np.array([None]*(N_coarse + 1))
            coarse_length_max = np.array([None]*(N_coarse + 1))
            coarse_speed_min  = np.array([None]*(N_coarse + 1))
            coarse_speed_max  = np.array([None]*(N_coarse + 1))
            coarse_length_min[0] = current_x_SV[i][0]
            coarse_length_max[0] = current_x_SV[i][0]
            coarse_speed_min[0]  = current_x_SV[i][1]
            coarse_speed_max[0]  = current_x_SV[i][1]
            G_i = np.zeros((4, 2*N))
            g_i = np.zeros((4, N))
            Reachable_Set_Reduced_i = list( )
            Reachable_Set_Reduced_i.append(current_x_SV[i])
            for t in range(1, N_coarse + 1):
                if t == 1:
                    reachable_set_reduced_t = (A_SV_coarse@Reachable_Set_Reduced_i[t - 1] + BU_SV_Reduced_Poly)
                else:
                    reachable_set_reduced_t = (A_SV_coarse*Reachable_Set_Reduced_i[t - 1] + BU_SV_Reduced_Poly) & X_SV_Poly
                Reachable_Set_Reduced_i.append(reachable_set_reduced_t)
                vertex = reachable_set_reduced_t.V
                vertex_x = vertex[:, 0]
                vertex_v = vertex[:, 1]
                coarse_length_min[t] = np.min(vertex_x)
                coarse_length_max[t] = np.max(vertex_x)
        
            coarse_interval = np.linspace(0, N_coarse*T_coarse, N_coarse + 1)
            fx_min = interpolate.interp1d(coarse_interval, coarse_length_min, kind = 'quadratic')
            fx_max = interpolate.interp1d(coarse_interval, coarse_length_max, kind = 'quadratic')
            fine_interval = np.linspace(0, N*T, N + 1)
            fine_length_min = fx_min(fine_interval)
            fine_length_max = fx_max(fine_interval)
            Robust_SV_Position_i = np.ones((4, N)) # leading vehicle with vehicle shape expanded, middle_x, middle_y, dx, dy
            
            for t in range(1, N + 1):
                if fine_length_min[t] > fine_length_max[t]:
                    min_x = (fine_length_min[t] + fine_length_max[t])/2 - l_veh
                    max_x = (fine_length_min[t] + fine_length_max[t])/2 + l_veh
                else:
                    min_x = fine_length_min[t] - l_veh
                    max_x = fine_length_max[t] + l_veh
                min_y = 1.5*w_lane - w_veh
                max_y = 2*w_lane
                temp_poly = Polytope(np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]]))
                G_i[:, 2*t-2:2*t] = temp_poly.A
                g_i[:, t-1]     = temp_poly.b.reshape(4, )
                Robust_SV_Position_i[:, t-1] = np.array([(max_x + min_x)/2, 1.5*w_lane, (max_x - min_x)/2, w_veh])
            
            Reachable_Set_Reduced.append(Reachable_Set_Reduced_i)
            G.append(G_i)
            g.append(g_i)
            Robust_SV_Position.append(Robust_SV_Position_i)
        
        return G, g, Robust_SV_Position
    
    def VevicleFrame2PointMass(self, vehicle_frame_state):
        DPM = self.DPM
        dimension = vehicle_frame_state.ndim
        if dimension == 1: 
            global_frame_state = np.array([vehicle_frame_state[0], 
                                           vehicle_frame_state[3]*np.cos(vehicle_frame_state[2]), 
                                           vehicle_frame_state[4]*np.cos(vehicle_frame_state[2]), 
                                           vehicle_frame_state[1],
                                           vehicle_frame_state[3]*np.sin(vehicle_frame_state[2]), 
                                           vehicle_frame_state[4]*np.sin(vehicle_frame_state[2])])
        else:
            n = vehicle_frame_state.shape[1]
            global_frame_state = np.zeros((DPM, n))
            for i in range(n):
                global_frame_state[:, i] = np.array([vehicle_frame_state[0], 
                                                     vehicle_frame_state[3]*np.cos(vehicle_frame_state[2]), 
                                                     vehicle_frame_state[4]*np.cos(vehicle_frame_state[2]), 
                                                     vehicle_frame_state[1],
                                                     vehicle_frame_state[3]*np.sin(vehicle_frame_state[2]), 
                                                     vehicle_frame_state[4]*np.sin(vehicle_frame_state[2])])
        return global_frame_state
    
    def Return(self, current_x_SV, current_y_SV, current_x_EV, samples):
        N_M = self.N_M
        T   = self.T
        N   = self.N
        L_Center = self.L_Center
        L_Bound  = self.L_Bound
        N_Lane   = self.N_Lane
        SpeedNom = self.SpeedNom
        Weight   = self.Weight
        road_terminal = self.road_terminal
        l_veh         = self.l_veh
        
        EV_point_mass_k = self.VevicleFrame2PointMass(current_x_EV)
        RefPrim = [EV_point_mass_k[1], EV_point_mass_k[1]]
        
        # update the primary reference accroding to road speed limit
        for i in range(N_Lane):
            if (np.sum(SpeedNom[i]) != None) and (np.sum(RefPrim[i]) != None): # the lane has nominal speed and the state is not empty 
                RefPrim[i] = SpeedNom[i]
                
        G, g, Robust_SV_Position_k = self.ReachableSet(current_x_SV, current_y_SV, samples)
        REF = self.ProjectSpeed(Robust_SV_Position_k, current_x_SV, EV_point_mass_k, RefPrim) 
        # calculate the action cost and model probability
        ActPse = np.array([None]*N_M)
        t = np.arange(0, T*(N + 1), T, dtype = float)
        L = np.array([None]*N_M)
        c = np.array([1]*N_M)
        
        if current_x_EV[0] + l_veh/2  < road_terminal[0]:
            for i in range(N_M): 
                X = self.VelocityTracking(EV_point_mass_k, REF[i], i, N)
                ax = X[2, :]*X[2, :]
                ay = X[5, :]*X[5, :]
                ActPse[i] = Weight[0]*np.trapz(ax, t) + Weight[1]*np.trapz(ay, t) + \
                Weight[2]*((REF[i] - EV_point_mass_k[1])**2) + Weight[3]*((L_Center[i] - EV_point_mass_k[3])**2)
                ActPse[i] = ActPse[i] + 0.0001
                L[i] = 1/np.sqrt(ActPse[i])
        else:
            L[0], L[1] = 0, 1

        temp = c@L
        mu_k = c*L/temp
        m_k = np.argmax(mu_k)
        RefSpeed = REF[m_k]
        RefLane = L_Center[m_k]
        if EV_point_mass_k[0] < current_x_SV[1][0]:
            RefSpeed = REF[1]
            RefLane = L_Center[1]
            mu_k = np.array([0, 1])
            
        if np.abs(current_x_EV[1] - L_Center[1]) <= 0.75:
            Lane_low = L_Bound[1] + 1
        else:
            Lane_low = L_Bound[0] + 1
        Trajectory_k, U_k = self.MPCFormulation(G[0], g[0], G[1], g[1], current_x_EV, RefSpeed, RefLane, Lane_low)
        Trajectory_k      = Trajectory_k.full( )
        U_k               = U_k.full( )
        
        return U_k[:, 0], Trajectory_k, REF, RefSpeed, RefLane, Robust_SV_Position_k, mu_k
    
    def Ref_Speed_Stage(self):
        N = self.N
        d_min = self.d_min

        opti      = casadi.Opti( )
        X_lim_low = opti.parameter(N, 1)
        X_lim_up  = opti.parameter(N, 1)
        A_x       = opti.parameter(N, 1)
        B_x       = opti.parameter(N, 1)
        A_v       = opti.parameter(N, 1)
        B_v       = opti.parameter(N, 1)
        v_pri     = opti.parameter( )

        v_up = opti.variable( )
        X_EV = A_x*v_up + B_x
        J = (v_pri - v_up)**2
        
        opti.minimize(J)
        opti.subject_to((X_lim_low + d_min) <= X_EV)
        opti.subject_to((X_EV) <= (X_lim_up - d_min))
        opts = {"ipopt.print_level": 0,"ipopt.linear_solver": "ma57","print_time": False}
        opti.solver('ipopt', opts)
        
        return opti.to_function('f', [X_lim_low, X_lim_up, A_v, B_v, A_x, B_x, v_pri], [v_up])
    
    def MPCFormulation(self):
        N         = self.N
        Np        = self.Np
        DEV       = self.DEV
        d_min     = self.d_min_mpc
        w_lane    = self.w_lane
        T         = self.T
        Q1        = self.Q1
        Q2        = self.Q2
        Q3        = self.Q3
        Q4        = self.Q4
        v_low     = self.v_low 
        v_up      = self.v_up 
        acc_low   = self.acc_low 
        acc_up    = self.acc_up 
        delta_low = self.delta_low 
        delta_up  = self.delta_up

        opti = casadi.Opti( )
        X       = opti.variable(DEV, Np + 1)
        U       = opti.variable(2, Np)
        lam_0   = opti.variable(4, Np)
        lam_1   = opti.variable(4, Np)
        delta   = U[0, :]
        eta     = U[1, :]

        G_0      = opti.parameter(4, 2*N)
        g_0      = opti.parameter(4, N)
        G_1      = opti.parameter(4, 2*N)
        g_1      = opti.parameter(4, N)
        Initial  = opti.parameter(DEV, 1)
        v_ref    = opti.parameter( )
        y_ref    = opti.parameter( )
        Lane_low = opti.parameter( )
        
        opti.subject_to(X[:, 0] == Initial)
        for k in range(Np):
            k1 = self.vehicle_model(X[:, k], delta[k], eta[k])
            k2 = self.vehicle_model(X[:, k] + T/2*k1, delta[k], eta[k])
            k3 = self.vehicle_model(X[:, k] + T/2*k2, delta[k], eta[k])
            k4 = self.vehicle_model(X[:, k] + T*k3, delta[k], eta[k])
            x_next = X[:, k] + T/6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X[:, k + 1] == x_next) 
        
        y = X[1, 1::]
        v = X[3, 1::]
        a = X[4, 1::]
        y_error = y[-1] - y_ref 
        v_error = v[-1] - v_ref 
        for k in range(Np):
            p_point = X[0:2, k + 1]
            
            G_0_point = G_0[:, 2*k:2*k + 2]
            g_0_point = g_0[:, k]
            temp_0 = G_0_point.T@lam_0[:, k]
            
            G_1_point = G_1[:, 2*k:2*k + 2]
            g_1_point = g_1[:, k]
            temp_1 = G_1_point.T@lam_1[:, k]
            
            opti.subject_to((G_0_point@p_point - g_0_point).T@lam_0[:, k] >= d_min)
            opti.subject_to((temp_0[0]**2 + temp_0[1]**2) <= 1)
            opti.subject_to(0 <= lam_0[:, k])
            
            opti.subject_to((G_1_point@p_point - g_1_point).T@lam_1[:, k] >= d_min)
            opti.subject_to((temp_1[0]**2 + temp_1[1]**2) <= 1)
            opti.subject_to(0 <= lam_1[:, k])
            
        opti.subject_to(opti.bounded(Lane_low, y, 2*w_lane - 1))
        opti.subject_to(opti.bounded(v_low, v, v_up))
        opti.subject_to(opti.bounded(acc_low, a, acc_up))
        opti.subject_to(opti.bounded(delta_low, delta, delta_up))
        
        J = delta@Q1@delta.T + eta@Q2@eta.T + y_error@Q3@y_error.T + v_error@Q4@v_error.T
        opti.minimize(J)
        
        opts = {"ipopt.print_level": 0, "ipopt.linear_solver": "ma57", "print_time": False}
        opti.solver('ipopt', opts)

        return opti.to_function('g', [G_0, g_0, G_1, g_1, Initial, v_ref, y_ref, Lane_low], [X, U])

    def  vehicle_model(self, w, delta, eta):
        l_f = self.l_f
        l_r = self.l_r

        x_dot   = w[3] 
        y_dot   = w[3]*w[2] + (l_r/(l_f + l_r))*w[3]*delta
        phi_dot = w[3]/(l_f + l_r)*delta
        v_dot   = w[4]
        a_dot   = eta
        
        return casadi.vertcat(x_dot, y_dot, phi_dot, v_dot, a_dot)