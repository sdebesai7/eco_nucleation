import numpy as np
import scipy 
import pickle as pkl

#simulation to run Lotka-Volterra Dynamics with migration from a species global pool
#paramters: 
#m: per capita migration rate
#N: total population size 
#mu_i: species relative per capita migraiton rate
#N_i: species i population 
#T_tot: number of events simulated
#T: vector of migration events simulated
#S: number of species in the global species pool
#A: SxS matrix

def init_sim(S, T_tot, m, mu, max_r):
   N=np.zeros(S)
   delta_T=np.random.exponential(scale=m, size=T_tot)#generate mutation events
   rng = np.random.default_rng()
   #pick the order of arrivals
   migrants=rng.choice(S, size=T_tot, replace=True, p=mu)
       
   #A_ii=1 and A_ij is a randomly sampled number
   A=(np.random.uniform(-max_r, max_r, (S, S)) * abs(np.eye(S)-1)) - np.eye(S) 
   
   return A, delta_T, N, migrants

def LV_dynamics(t, N, A, rates):
    #solve LV dynamics 
    return np.multiply(np.multiply(rates, N), (np.ones(N.shape[0])-np.multiply(N, np.matmul(A, N))))

def grow(A, ti, tf, N, rates):
    sol=scipy.integrate.solve_ivp(LV_dynamics, t_span=(ti, tf), y0=N, args=(A, rates))
   
    return sol.y[:, -1]

def full_simulate(T_tot, S, N, num_migrants, rates, A, delta_T, migrants):
    N_v_t=np.zeros((T_tot, S))
    T=np.zeros(T_tot+1)
    for i, t in enumerate(T[:-1]):
        N_v_t[i]=N #2D array where A_ij=the count of species j at time i
        T[i+1]=T[i]+delta_T[i]
        #introduce migrant
        N[migrants[i]]=N[migrants[i]]+num_migrants
        #choose the number 
        N=grow(A, T[i], T[i+1], N, rates)
    
    return N_v_t, T