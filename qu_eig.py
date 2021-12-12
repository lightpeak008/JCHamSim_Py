import numpy as np



#---------- Constructing Kets
def fock(n, k):
    psi = np.zeros((n,1), dtype='c16')
    psi[k,0] = 1
    return psi


#---------- Constructing Operators
def identity(n):
    return np.identity(n, dtype='c16')
    
def zeros(n):
    return np.zeros((n,n), dtype='c16')
    
def destroy(n):
    offDiagVal = np.sqrt(range(1,n), dtype='c16')
    return np.diag(offDiagVal, 1)

def number(n):
    diagVal = np.arange(n, dtype='c16')
    return np.diag(diagVal, 0)
    

#---------- Constructing Operator Transformation
def dag(oper):
    return np.conj(np.transpose(oper))


#---------- Constructing Function for Expectation Values
def expect(oper, psi):
    return np.vdot(psi, oper.dot(psi))/np.vdot(psi, psi)

def expectAvg(oper, psi):
    avg = 0
    for i in range(psi.shape[1]):
        avg += abs(expect(oper, psi[:,i]))
    return avg/psi.shape[1]


#---------- rk4Solve Routine
def rk4Step(H, psi, t, c_op, delt, H_args):
    # Code designed for just one Time Dependent Hamiltonian
    H0 = H[0]
    
    H1, H1_coeff = H[1]
    H1_args = H_args[0]
    
    H0_c = H0 - 0.5j*dag(c_op).dot(c_op)
    c0 = H1_coeff(t, H1_args)
    c1 = H1_coeff(t+0.5*delt, H1_args)
    c2 = H1_coeff(t+delt, H1_args)
    
    k1 = delt*(-1j*(H0_c + c0*H1).dot(psi))
    k2 = delt*(-1j*(H0_c + c1*H1).dot(psi + 0.5*k1))
    k3 = delt*(-1j*(H0_c + c1*H1).dot(psi + 0.5*k2))
    k4 = delt*(-1j*(H0_c + c2*H1).dot(psi + k3))
    return psi + (k1 + 2*k2 + 2*k3 + k4)/6


def rk4Solve(H, psi0, cyclePeriod, cycleRes, cycleCount, c_op, e_op, H_args, ntraj, show_progress=True):
    psi = np.ones((psi0.shape[1], ntraj), dtype='c16')*psi0
    delt = cyclePeriod/cycleRes
    num_op = number(psi0.shape[0])
    
    data = np.zeros((cycleRes*cycleCount+1,3))
    data[:,0] = np.arange(data.shape[0])*delt
    data[0,1] = abs(expect(num_op, psi0))
    data[0,2] = abs(expect(e_op, psi0))
    
    delt = cyclePeriod/cycleRes
    
    limitFlag = False
    limitThreshold = 1E-10
    
    print('Running Simulation')
    p_rand = np.random.rand(ntraj)

    for i in range(cycleCount):
        for j in range(cycleRes):
            t = i*cyclePeriod + delt*j
            psi = rk4Step(H, psi, t, c_op, delt, H_args)
                
            p_norm = np.linalg.norm(psi, axis=0)**2
            
            if not limitFlag:
                limitTest = (abs(psi[-1,:])**2)/p_norm
                if np.any(limitTest > limitThreshold): limitFlag = True
            
            p_clist = p_norm < p_rand
            if np.any(p_clist):
                # Code designed for just one collapse operator
                p_rand[p_clist] = np.random.rand(np.count_nonzero(p_clist))
                psi[:,p_clist] = c_op.dot(psi[:,p_clist])
                psi[:,p_clist] = psi[:,p_clist]/np.linalg.norm(psi[:,p_clist], axis=0)

            data[i*cycleRes+j+1, 1] = expectAvg(num_op, psi)
            data[i*cycleRes+j+1, 2] = expectAvg(e_op, psi)
        
        if show_progress: print('Total Progress: %.1f%%' %((i+1)*100./cycleCount), end='\r')
    
    if limitFlag: print('\n\nWarning: Exceeded Truncated Hilbert Space--------------')    
    print('\nComplete')
    return data


#---------- Constructing Subroutine for eigSolve
def genStepOper(H, c_op, cyclePeriod, cycleRes, H_args, show_progress=True):
    # Code designed for just one Time Dependent Hamiltonian
    H0 = H[0]
    
    H1, H1_coeff = H[1]
    H1_args = H_args[0]
    
    
    delt = cyclePeriod/cycleRes
    operList = []
    print('Number of Steps per Cycle: %d' %cycleRes)    
    print('Computing Time Step Operators')
    for i in range(cycleRes):
        t = i*delt
        H = H0 - 0.5j*dag(c_op).dot(c_op) + H1_coeff(t, H1_args)*H1
        w, v = np.linalg.eig(H)
        rot = np.diag(np.exp(-1j*w*delt))
        vi = np.linalg.inv(v)
        operList.append(v.dot(rot.dot(vi)))
        if show_progress: print('Total Progress: %.1f%%' %((i+1)*100./cycleRes), end='\r')
    
    print('\nDone')
    return operList


#---------- eigSolve Routine
def eigSolve(H, psi0, cyclePeriod, cycleRes, cycleCount, c_op, e_op, H_args, ntraj, show_progress=True):
    psi = np.ones((psi0.shape[1], ntraj), dtype='c16')*psi0
    delt = cyclePeriod/cycleRes
    num_op = number(psi0.shape[0])
    
    data = np.zeros((cycleRes*cycleCount+1,3))
    data[:,0] = np.arange(data.shape[0])*delt
    data[0,1] = abs(expect(num_op, psi0))
    data[0,2] = abs(expect(e_op, psi0))
    
    stepOper = genStepOper(H, c_op, cyclePeriod, cycleRes, H_args, show_progress)
    
    limitFlag = False
    limitThreshold = 1E-10
    
    print('Running Simulation')
    p_rand = np.random.rand(ntraj)

    for i in range(cycleCount):
        for j in range(cycleRes):
            psi = stepOper[j].dot(psi)
                
            p_norm = np.linalg.norm(psi, axis=0)**2
            
            if not limitFlag:
                limitTest = (abs(psi[-1,:])**2)/p_norm
                if np.any(limitTest > limitThreshold): limitFlag = True
            
            p_clist = p_norm < p_rand
            if np.any(p_clist):
                # Code designed for just one collapse operator
                p_rand[p_clist] = np.random.rand(np.count_nonzero(p_clist))
                psi[:,p_clist] = c_op.dot(psi[:,p_clist])
                psi[:,p_clist] = psi[:,p_clist]/np.linalg.norm(psi[:,p_clist], axis=0)

            data[i*cycleRes+j+1, 1] = expectAvg(num_op, psi)
            data[i*cycleRes+j+1, 2] = expectAvg(e_op, psi)
        
        if show_progress: print('Total Progress: %.1f%%' %((i+1)*100./cycleCount), end='\r')
    
    if limitFlag: print('\n\nWarning: Exceeded Truncated Hilbert Space--------------')    
    print('\nComplete')
    return data

