# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 14:05:29 2016

@author: ilker bayram
"""

import numpy as np

def FProject(alp, y):
    # finds the vector bet closest to alp such that
    # bet >= 0, bet' * y = 0.
    #
    # input variables : 
    # alp : column vector of real numbers
    # y : class variables taking values in {1, -1}
    #
    # output variables : 
    # bet : sought vector
    alps = np.sort(alp * y)
    # check the minimum endpoint
    t = alps[0]
    s = np.sum( y * np.maximum( alp - y * t, 0) )
    if s < 0:
        n = np.sum( (alp - y * t) > 0 )
        t = s/float(n)
        bet = np.maximum( alp - y * t, 0)
        return bet
    
    # check the maximum endpoint
    t = alps[-1]
    s = np.sum( y * np.maximum( alp - y * t, 0) )
    if s > 0:
        n = np.sum( (alp - y * t) > 0 )
        t = s/float(n)
        bet = np.maximum( alp - y * t, 0)
        return bet
    
    # the best threshold lies somewhere in between the endpoints of alps

    indmin = 0
    indmax = alps.size-1
    while  (indmax - indmin) > 1 :
        K = np.floor( (indmin + indmax)/ 2.0 )
        K = int(K)
        t = alps[K] # the threshold
        s = np.sum( y * np.maximum( alp - y * t, 0) )
        
        if s > 0: # if s > 0, the sought threshold < t
            indmin = K
        elif s < 0: # if s < 0, the sought threshold > t
            indmax = K
        else: # if s = 0, t is the sought threshold
            bet = np.maximum( alp - y * t, 0)
            return bet
        
    # indmin and indmax are now determined
    # it remains to find the correct value of t that lies in between
    # alps[indmin] and alps[indmax]
    t = (alps[indmin] + alps[indmax]) / 2.0
    n = np.sum( (alp - y * t) > 0 ) # number of non-zeros
    dif = np.sum( y * np.maximum( alp - y * t, 0) ) / float(n)
    t = t + dif # update the threshold
    bet = np.maximum( alp - y * t, 0)
    return bet
    
def SVMdual_FB(H, y, MAX_ITER):
    # Forward backward splitting (FBS) algorithm for solving the quadratic programming problem
    # min_x  (1/2) <x,Hx> - <1,x> subject to <y,x> = 0, x_i >= 0
    # it is assumed that y_i takes values in {-1,1}
    #
    # input variables :
    # H : the matrix in the problem definition
    # y : the class vector in the problem definition
    # MAX_ITER : maximum number of iterations for the FBS algorithm
    #
    # output variable :
    # x : solution of the problem
    
    bet = max( sum( abs(H) ) ) # this is an upper bound for the spectral norm of H
    x = np.random.uniform(0,0.1,y.size)
    for iter in range(0,MAX_ITER):
        print '\r SVM training : [{0}{1}] {2}% Complete'.format('|'*(10 * (iter+1) / MAX_ITER), ' '*(10 - 10 * (iter+1) / MAX_ITER), 100 * (iter+1) / MAX_ITER),
        
        # the forward step
        x = x - (2.0/bet) * ( np.dot(H,x) - 1)
        
        # the backward (projection) step
        x = FProject( x, y)
    
    return x
    