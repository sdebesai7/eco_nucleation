
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