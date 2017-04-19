import warnings
import exceptions
import sys
import matplotlib.pylab as mp
import time
import numpy
import numpy as np
from vector import CGaussKernel,CLinearKernel,CRBFKernel
from numpy.random import randn,rand
import hungarian #,LAPJV # LAP solvers
import pdb
np.seterr(divide='ignore', invalid='ignore')

def KS(X_1,X_2):
    init_type = 'eig' # random, eig, ...
    llambda = 1.0 # step size
    n_iter = 100 # number of LAP iterations
    omegas = 2.0
    n_obs = X_1.shape[0] # number of observations
    bases = numpy.eye(n_obs) 
    # normalization of data
    l2norm = numpy.zeros((n_obs,1))
    for i in xrange(n_obs):
        l2norm[i] = numpy.sum(X_1[i,])
    new_X_1 = X_1/l2norm
    
    starttime = time.clock()
    # kernel for grid
    dl = CRBFKernel();
    dL = dl.Dot(X_2, X_2)
    omega_L = 1.0 / numpy.median(dL.flatten())	
    kernel_L = CGaussKernel(omega_L)    
    L = kernel_L.Dot(X_2,X_2) # should use incomplete cholesky instead ...
    # kernel for data
    tK = numpy.zeros((n_obs,n_obs))
    for i in xrange(n_obs):
        for j in xrange(i,n_obs):
            temp = ((new_X_1[i,:]-new_X_1[j,:])**2)/(new_X_1[i,:]+new_X_1[j,:])
            inan = numpy.isnan(temp)
            temp[inan] = 0.0
            tK[i,j] = tK[j,i] = numpy.sum(temp)*0.5
    mdisK = numpy.median(numpy.median(tK))
    K = numpy.zeros((n_obs,n_obs))
    for i in xrange(n_obs):
        for j in xrange (i,n_obs):
            temp = ((new_X_1[i,:]-new_X_1[j,:])**2)/(new_X_1[i,:]+new_X_1[j,:])
            inan = numpy.isnan(temp)
            temp[inan] = 0.0
            K[i,j] = K [j,i] = numpy.exp(-1.0*omegas/mdisK*numpy.sum(temp)*0.5) #chi-square kernel
    
    stoptime = time.clock()
    print 'computing kernel matrices (K and L) takes %f seconds '% (stoptime-starttime)   
        
    print 'original objective function : '
    H = numpy.eye(n_obs) - numpy.ones(n_obs)/n_obs
    print numpy.trace(numpy.dot(numpy.dot(numpy.dot(H,K),H),L))      
    
    # initializing the permutation matrix
    if (cmp(init_type,'random') == 0):
        print 'random initialization is being used ... \n'
        PI_0 = init_random(n_obs)
    elif (cmp(init_type,'eig') == 0):
        print 'sorted eigenvector initialization is being used ... \n'
        PI_0 = init_eig(K,L,n_obs)
    else:
        print 'wrong initialization type ... '

    # centering of kernel matrices    
    H = numpy.eye(n_obs) - numpy.ones(n_obs)/n_obs
    K = numpy.dot(H,numpy.dot(K,H))
    L = numpy.dot(H,numpy.dot(L,H))

    print 'initial objective: '
    print numpy.trace(numpy.dot(numpy.dot(numpy.dot(PI_0,K),PI_0.T),L))      

    # iterative linear assignment solution
    PI_t = numpy.zeros((n_obs,n_obs))
    for i in xrange(n_iter):
        print 'iteration : ',i
        starttime = time.clock()        
        grad = compute_gradient(K,L,PI_0) # L * P_0 * K
        stoptime = time.clock()
        print 'computing gradient takes %f seconds '% (stoptime-starttime)
        
        # convert grad (profit matrix) to cost matrix
        # assuming it is a finite cost problem, thus 
        # all the elements are substracted from the max value of the whole matrix
        cost_matrix = -grad
        starttime = time.clock()             
        #indexes = LAPJV.lap(cost_matrix)[1]
        indexes = hungarian.hungarian(cost_matrix)[0]
        indexes = numpy.array(indexes)
        
        stoptime = time.clock()
        print 'lap solver takes %f seconds '% (stoptime-starttime)
        
        PI_t = numpy.eye(n_obs)
        PI_t = PI_t[indexes,]
        
        # convex combination
        PI = (1-llambda)*PI_0 + (llambda)*PI_t
        # gradient ascent
        #PI = PI_0 + llambda*compute_gradient(K,L,PI_t)

        # computing the objective function
        obj_funct = numpy.trace(numpy.dot(numpy.dot(numpy.dot(PI_t,K),PI_t.T),L))
        print 'objective function value : ',obj_funct
        print '\n'

        # another termination criteria
        if (numpy.trace(numpy.dot(numpy.dot(numpy.dot(PI,K),PI.T),L)) - numpy.trace(numpy.dot(numpy.dot(numpy.dot(PI_0,K),PI_0.T),L)) <= 1e-5):
                PI_final = PI_t
                break
      
        PI_0 = PI
        if (i == n_iter-1):
            PI_final = PI_t
            
    return PI_final
def compute_gradient(K,L,PI_0):
    grad = numpy.dot(L,numpy.dot(PI_0,K))
    grad = 2*grad
    return grad

def init_eig(K,L,n_obs):
    # with sorted eigenvectors
    [U_K,V_K] = numpy.linalg.eig(K)
    [U_L,V_L] = numpy.linalg.eig(L)
    i_VK = numpy.argsort(-V_K[:,0])
    i_VL = numpy.argsort(-V_L[:,0])
    PI_0 = numpy.zeros((n_obs,n_obs))
    PI_0[numpy.array(i_VL),numpy.array(i_VK)] = 1
    return PI_0

def init_random(n_obs):
    # random initialization
    bases = numpy.eye(n_obs)    
    init = numpy.random.permutation(n_obs)
    PI_0 = bases[init,:]
    return PI_0
       

if __name__=='__main__':
    main()