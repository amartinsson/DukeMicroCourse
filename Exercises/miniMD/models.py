#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 16:51:43 2018

@author: amartinsson, msachs2 
"""
import numpy as np
import abc


class Model(object):
    """ Base class which must implement get_force and get_potential
    functions.
    """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, dim):
        self._dim = dim
        
    def get_force(self, position):
        raise NotImplementedError()
        
    def get_potential(self, position):
        raise NotImplementedError()

    def update_force(self):
        """ updates the force internally from the current position
        """
        self.f = self.get_force(self.q)

    def apply_boundary_conditions(self):
        raise NotImplementedError()
    
class Particle(Model):
    """ Class which holds data to do molecular dynamics
    sampling with.
    """
    def __init__(self, q, p=None):
        """ Init function for this class
        
        :param q: position
        :param p: momentum
        :param f: force
        :param m: mass
        """
        super(Particle, self).__init__(q.size)

        # set the current position
        self.q = q

        # update the force based on the
        # current positon
        self.f = q
        self.update_force()

        # set the momentum if it's found
        self.p = p
        
    def get_q_asVec(self):
        return self.q
    
    def get_p_asVec(self):
        return self.p
    
    def apply_boundary_conditions(self):
        pass

class ReplicatedModel(Model):
    """ Baseclass for models used in samplers using multiple replicas.
    """    
    
    def __init__(self, model, nreplicas=1):
        self._model = model
        self._dim = model._dim * nreplicas
        self._nreplicas = nreplicas
        self.q = np.repeat(np.reshape(model.q, (-1, model._dim)),self._nreplicas, axis=0)
        self.f = np.repeat(np.reshape(model.f, (-1, model._dim)),self._nreplicas, axis=0)
        if model.p is not None:
            self.p = np.repeat(np.reshape(model.p, (-1, model._dim)),self._nreplicas, axis=0)
        
    def get_potential(self):
        """ returns the potential at the current position
        """
        pot = np.zeros(self.nreplicas)
        for i in range(self.nreplicas):
            pot[i] = self._model.get_potential(self.q[i,:])
            
        return pot

    def update_force(self):
        self.f = self.get_force(self.q)
        
    def get_force(self,q):
        """ updates the force internally from the current position
        """
        f = np.zeros([self._nreplicas, self._model._dim])
        for i in range(self._nreplicas):
            f[i,:] = self._model.get_force(q[i,:])
            
        return f
    
    def get_q_asVec(self):
        return self.q.flatten()
    
    def get_p_asVec(self):
        return self.p.flatten()
    
    
    def apply_boundary_conditions(self):
        pass # replicated model is assumed to have no speficied boundary conditions
    
    def collapse_traj(self, q_trajectory):
        T = q_trajectory.shape[0]
        collapsed_traj = np.zeros([T*self._nreplicas, self._model._dim])
        for i in range(self._nreplicas):
            collapsed_traj[i*T:(i+1)*T, :] = q_trajectory[:, i*self._model._dim:(i+1)*self._model._dim]
        return collapsed_traj
    
class HarmonicOscillatorMultiDim(Particle):
    
    def __init__(self, q, p=None, Omega=None ):
        
        if Omega is None:
            Omega = np.eye(self._dim)
        
        self.Omega = Omega
        
        super(HarmonicOscillatorMultiDim, self).__init__(q,p)
        
        
        
    def get_potential(self):
        """ returns the potential at the current position
        """
        return .5 * np.dot(self.q,np.dot(self.Omega, self.q))

    def get_force(self, q):
        """ returns the force internally from the current position
        """
        return -np.dot(self.Omega, q)
    
    
    def apply_boundary_conditions(self):
        pass
    
class HarmonicOscillator(Particle):
    """ Class which implements the force and potential for the
    harmonic oscillator particle model.
    """    
    def get_potential(self):
        """ returns the potential at the current position
        """
        return .5 * self.q * self.q

    def get_force(self,q):
        """ returns the force internally from the current position
        """
        return -q
    
class CosineModel(Particle):
    """ Class which implements the force and potential for the cosine model
    """
    def __init__(self, q, p=None, L=2.*np.pi):
        """ Init function for the class
        :param q: position
        :param p: momentum
        :param f: force
        :param L: length of periodic box
        """
        super(CosineModel, self).__init__(q,p)

        # Length of the periodic box
        self._L = L
    
    def get_force(self,q):
        """ updates the force internally from the current position
        """
        return np.sin(q)
        
    def get_potential(self):
        """ returns the potential at the current position
        """
        return np.cos(self.q)

    def apply_boundary_conditions(self):
        self.q = np.mod(self.q,self._L)

class BayesianLogisticRegression(Particle):
    
    def __init__(self, prior_mean, prior_cov, data, q, p=None ):
        
        
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.data = data
        
        self.prior_cov_inv = np.linalg.inv(self.prior_cov)
        self.prior_cov_det = np.linalg.det(self.prior_cov)
        
        super(BayesianLogisticRegression, self).__init__(q,p)
        
    #def as2Dvector(self, position):
     #   return np.reshape(position, (-1, self._dim))
    
    def get_force(self, position):
        return self.grad_log_like(position) + self.grad_log_prior(position)
        
    def get_potential(self, position):
        return self.log_like(position) + self.log_prior(position)
        
    
    def predict(self, params, x):
        return self._logistic(np.dot(x, params))
        
    def log_like(self, params):
        x, y = self.data
        return np.sum(self.predict(params, x))
    
    def grad_log_like(self, params):
        x, y = self.data
        prob0 = self.predict(params, x)
        return np.dot( (y - prob0), x)
    
    def log_prior(self, x):
        return -np.dot(x - self.prior_mean, np.dot(self.prior_cov_inv, x - self.prior_mean))/2.0
        
    def grad_log_prior(self, x):
        return -np.dot(self.prior_cov_inv, x - self.prior_mean)
    
    def _logistic(self, x):
        expx = np.exp(x)
        prob0 = np.zeros(x.shape)
        index = np.isinf(expx)
        prob0[np.logical_not(index)] = expx[np.logical_not(index)]/(1.0+expx[np.logical_not(index)])
        prob0[index] = 1.0
        
        return prob0
    
    def predict_from_sample(self, params_traj, x): # params_traj  is a T \times self.dim  matrix, where each column represents one sample of params 
        '''
        returns the probability vector 1/T sum_{t=0}^{T-1}[P(y_i| params[:,t], x)]_{ i = 0, ..., n_classes-1}  
        '''
        T = params_traj.shape[0]
        prob = np.zeros(2)
        for t in range(T):
            params = params_traj[t,:]            
            prob += self.predict(params, x)
        prob /= T
        
        return prob   
    
    def plot_prediction(self, q_trajectory, grid, Neval=100, show_training_data=True ):
        import time
        import matplotlib.pyplot as plt
        
        xx1, xx2 = grid
        t = time.time()
        z= np.zeros([len(xx1),len(xx2)])
        modthin = q_trajectory.shape[0]//Neval
        for i in range(len(xx1)):
            for j in  range(len(xx2)):
                x = np.array([xx1[i],xx2[j]])
                z[i,j] = self.predict_from_sample(q_trajectory[::modthin,:self._dim], x)[0]

        elapsed = time.time() - t
        
        print('Time to calculate predictions: {}'.format(elapsed))

        fig2, ax2 = plt.subplots()
        ax2.pcolor(xx1, xx2, z.transpose(), cmap='RdBu', vmin=0, vmax=1)
        cax = ax2.pcolor(xx1, xx2, z.transpose(), cmap='RdBu', vmin=0, vmax=1)
        cbar = fig2.colorbar(cax)
        cbar.ax.set_ylabel('$\mathbb{P}(Y = 1)$')
    
        if show_training_data:
            '''
            Include training data
            '''
            X,Y = self.data
            ndata = Y.shape[0]
            color_dict= {0:'red', 1 :'blue'}
            colors = [color_dict[Y[i]] for i in range(ndata)]
            
            ax2.scatter(X[:,0],X[:,1], c=colors)
        
        ax2.set_title('Prediction')
        ax2.set_xlabel('$\beta_1$')
        ax2.set_ylabel('$\beta_2$')
        return fig2, ax2