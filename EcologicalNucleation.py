import numpy as np
import scipy 
import pickle as pkl
import math

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

class EcoSim():
    def __init__(self, N, S, m, T_tot, mu, A, rates, num_migrants, curr_lambda):
        self.N=N 
        self.S=S 
        self.T_tot=T_tot #this is in units of migration events
        self.m=m
        self.mu=mu
        rng = np.random.default_rng()
        self.delta_T=np.random.exponential(scale=m, size=self.T_tot)#generate mutation events
        #pick the order of arrivals
        self.migrants=rng.choice(S, size=self.T_tot, replace=True, p=mu)
        #A_ii=1 and A_ij is a randomly sampled number
        self.A=A
        self.rates=rates
        self.N_v_t=np.zeros((self.T_tot, self.S))
        self.T=np.zeros(self.T_tot+1)
        self.all_times=np.array([])
        self.all_solns=np.array([self.N])
        self.num_migrants=num_migrants
        self.curr_lambda=curr_lambda

    def LV_dynamics(self, t, x):
        print(np.sum(x))
        return np.multiply(x, self.rates + np.matmul(self.A, x)) #dx_i/dt = x_i(r + Ax)_i
    
    def grow(self, ti, tf):
        #stopping conditions
        def lambda_reached(t, x):
            return np.sum(x) - self.curr_lambda - 0.001 #stops when pop size == lambda

        def zero_reached(t, x):
            return np.sum(x) #stops when pop size == 0
    
        lambda_reached.terminal=True
        zero_reached.terminal=True

        sol=scipy.integrate.solve_ivp(self.LV_dynamics, t_span=(ti, tf), y0=self.N, dense_output=True, events=[lambda_reached, zero_reached], max_step=0.0001)
        return sol.t, sol.y, sol.y[:, -1], sol.t_events, sol.y_events
    
    def full_simulate(self):
        #simulate until you hit the next population size

        i=0
        
        while (np.sum(self.N) < self.curr_lambda) and (np.sum(self.N) > 0):
  
            self.N_v_t[i%self.T_tot]=self.N #2D array where A_ij=the count of species j at time i
            #delta_T=np.random.exponential(scale=self.m, size=1)
            migrant = self.migrants[i]
            
            
            self.N[migrant]=self.N[migrant]+self.num_migrants #increment migrant pop

            self.T[i+1]=self.T[i]+self.delta_T[i] #update time

            #growth subject to LV dynamics
            times, solns, self.N, t_events, y_events=self.grow(self.T[i], self.T[i+1])
            
            #print(np.sum(self.N))
            self.all_times=np.append(self.all_times, times) 
            self.all_solns=np.vstack((self.all_solns, solns.T))
            i=i+1
     
            #if we do more migration events than the max, we re-generate more random numbers 
            if i%self.T_tot==0 and i > 0:
                rng = np.random.default_rng()
                self.delta_T=np.concatenate((self.delta_T, np.random.exponential(scale=self.m, size=self.T_tot)))
                self.migrants=np.concatenate((self.migrants, rng.choice(self.S, size=self.T_tot, replace=True, p=self.mu)))
                self.T=np.concatenate((self.T, np.zeros(self.T_tot)))
        
        if np.sum(self.N) < self.curr_lambda and np.sum(self.N) > 0:
            success=True
        else:
            success=False
        
        return self.N_v_t, self.T, self.all_times, self.all_solns, i, success, t_events, y_events

#FFS wrapper

class FFS():
    def __init__(self, lambdas, S, num_trajs, A, m, mu, rates, T_tot):
        self.lambdas=lambdas #array of lambdas
        self.S=S #number of species
        self.num_trajs=num_trajs #number of trajectories to launch from each interface
        
        self.T_tot=T_tot #max time for each simulation
        self.A=A #interaction matrix
        self.m=m #relative migration rates
        self.mu=mu #total migration rate
        self.rates=rates #individual growth rates
        self.N_curr_lambda= np.empty((self.num_trajs, self.S+4))#2D array: num_trajs x num_species + 4
        self.N_prev_lambda= np.empty((self.num_trajs, self.S+4))#2D array: num_trajs x num_species + 4
        
        self.N_prev_lambda[:]=0 #initialize previous to 0, except success column is 1
        self.N_prev_lambda[:, -1]=1
    
    def next_lambda(self, curr_lambda):
        init_idx_choices=np.where(self.N_prev_lambda[:, -1] == 1) #get the possible initial indexes

        rng = np.random.default_rng()
        init_idx_choices=rng.choice(init_idx_choices, size=self.num_trajs, replace=True) #sample uniformly from these possible iniitial states

        #iterate through the trajs
        for t, idx in enumerate(init_idx_choices):
            N=self.N_prev_lambda[t][0:self.S-1] #get the initial configuration
            es=EcoSim(N, self.S, self.m, self.T_tot, self.mu, self.A, self.rates, curr_lambda) #instantiate a simulation
            N_v_t, T, all_times, all_solns, num_mig_events, success, t_events, y_events=es.full_simulate() #run the simulation
            self.N_prev_lambda[t]=np.append(N_v_t[-1], np.array([N_v_t, np.sum(N_v_t), idx, num_mig_events, success])) #update previous lambda array

    #dump the prev lambda data
    def save_state(self, curr_lambda):
        with open(f'{curr_lambda}_data.pkl', 'wb') as f:
            pkl.dump(self.N_prev_lambda, f)
            f.close()

    def run_FFS(self):
        for curr_lambda in self.lambdas: #iterate through every lambda
            self.save_state(curr_lambda) #save the previous lambda
            self.next_lambda(curr_lambda) #sample for next lambda

'''
        for i, t in enumerate(self.T[:-1]):
            self.N_v_t[i]=self.N #2D array where A_ij=the count of species j at time i
            self.T[i+1]=self.T[i]+self.delta_T[i]
            #introduce migrant
            self.N[self.migrants[i]]=self.N[self.migrants[i]]+self.num_migrants
            #choose the number 
            times, solns, self.N=self.grow(self.T[i], self.T[i+1])
            self.all_times=np.append(self.all_times, times)
            self.all_solns=np.vstack((self.all_solns, solns.T))
        '''
'''
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
    return np.multiply(N, rates + np.matmul(A, N))
    #np.array([rates[0]*N[0]*A[0, 0] + N[0]*N[1]*A[0, 1], rates[1]*N[1]*A[1, 0] + N[0]*N[1]*A[1, 1]])
    #np.multiply(np.multiply(rates, N), (np.ones(N.shape[0])-np.multiply(N, np.matmul(A, N))))

def grow(A, ti, tf, N, rates):
    sol=scipy.integrate.solve_ivp(LV_dynamics, t_span=(ti, tf), y0=N, args=(A, rates), dense_output=True)
   
    return sol.t, sol.y, sol.y[:, -1]

def full_simulate(T_tot, S, N, num_migrants, rates, A, delta_T, migrants):
    N_v_t=np.zeros((T_tot, S))
    T=np.zeros(T_tot+1)
    all_times=np.array([])
    all_solns=np.array([N])

    for i, t in enumerate(T[:-1]):
        N_v_t[i]=N #2D array where A_ij=the count of species j at time i
        T[i+1]=T[i]+delta_T[i]
        #introduce migrant
        N[migrants[i]]=N[migrants[i]]+num_migrants
        #choose the number 
        times, solns, N=grow(A, T[i], T[i+1], N, rates)
        all_times=np.append(all_times, times)

        all_solns=np.vstack((all_solns, solns.T))
    
    return N_v_t, T, all_times, all_solns
'''



