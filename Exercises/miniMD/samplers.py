#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 21:12:39 2018

@author: msachs2
"""

import abc
import numpy as np
import miniMD.models as models

class Sampler(object):
    """ Abstract base class for sampler integrator objects
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, model, stepsize):
        """ Init function for the class
        :param stepsize: discrete stepsize for time
        :param molecule: The molecule object which is sampled
        """
        self._stepsize = stepsize 
        self._model = model

    def sample(self,
               nsteps,
               outputmode="trajectory",
               qbins=1,
               pbins=2):
        
        """ Function which samples the molecule using the integrator
        object from the init function.
        
        :param nsteps: number of timesteps to perform the sampling
        :param outputmode: defines the form of the output which is returned
        :param qbins: defines the histogram bins of the q_trajectory
        :param pbins: defines the histogram bins of the p_trajectory
        """
        
        if outputmode is "trajectory":
            traj_q = np.zeros([nsteps,self._model._dim])
            traj_q[0,:] = self._model.get_q_asVec()
            
            if self._model.p is not None:
                traj_p = np.zeros([nsteps,self._model._dim])
                traj_p[0,:] = self._model.get_p_asVec()
            
            for t in range(nsteps):

                # step forward in time
                self.integrate()

                # collect the trajectory
                traj_q[t,:] = self._model.get_q_asVec()
                    
                if self._model.p is not None:
                    traj_p[t,:] = self._model.get_p_asVec()

            if self._model.p is not None:
                return traj_q, traj_p
            else:
                return traj_q

        elif outputmode is "cumulative_histogram":
            # initialise the histograms
            self.Hq = np.zeros(qbins.size-1)
            
            if self._model.p is not None:
                self.Hp = np.zeros(pbins.size-1)

            for t in range(nsteps):
                # step forward in time
                self.integrate()

                self.Hq += np.histogram(self._model.q,bins=qbins)[0]
            
                if self._model.p is not None:
                    self.Hp += np.histogram(self._model.p,bins=pbins)[0]

            if self._model.p is not None:
                return self.Hq, self.Hp
            else:
                return self.Hq

        elif outputmode is "histogram":
            # initialise the histograms
            self.Hq = np.zeros(qbins.size-1)
            
            if self._model.p is not None:
                self.Hp = np.zeros(pbins.size-1)

            for t in range(nsteps):
                # step forward in time
                self.integrate()

            self.Hq = np.histogram(self._model.q,bins=qbins)[0]
            
            if self._model.p is not None:
                self.Hp = np.histogram(self._model.p,bins=pbins)[0]

            if self._model.p is not None:
                return self.Hq, self.Hp
            else:
                return self.Hq
        else:
            raise ValueError("Outputmode not recognised")

# First order Dynamics Langevin Integrators
class OverDampedLangevinIntegrator(Sampler):
    """ Base class for all integrators which implements
    first order dynamics i.e no momentum evolution
    """

    def __init__(self,model, stepsize, inverse_temperature):
        """ Init function for the class

        :param stepsize: discrete stepsize for time
        :param model: class which is used to update the force
        :param inverse_temperature: scaling for the gradient - beta
        """
        super(OverDampedLangevinIntegrator, self).__init__(model, stepsize)

        self._inverse_temperature = inverse_temperature

    def integrate(self):
        """ Function which implements the integration forward in time
        for the molecule.
        
        :param molecule: object of type molecule which holds data
        """
        raise NotImplementedError()

# The Leimkuhler-Matthews method
class LeimkuhlerMatthews(OverDampedLangevinIntegrator):
    """ Class which implements the Leimkuhler-Matthews method
    which is a first order Integrator
    """

    def __init__(self,model,stepsize, inverse_temperature):
        """ Init function for the class

        :param stepsize: discrete stepsize for time
        :param model: class which is used to update the force
        :param inverse_temperature: scaling for the gradient - beta
        """
        if model.p is not None:
            raise ValueError('LeimkuhlerMatthews is only first Order Dynamics!')
        
        super(LeimkuhlerMatthews, self).__init__(model, stepsize, inverse_temperature)

        self._noise_k1 = np.random.normal(0., 1., self._model._dim)
        self._zeta = np.sqrt(.5 * self._stepsize / self._inverse_temperature )

    def integrate(self):
        """ Integration function which evolves the model
        given in self._model
        """
        noise_kp1 = np.random.normal(0., 1., self._model._dim)

        # pre force update
        self._model.q += self._stepsize * self._model.f + self._zeta * (self._noise_k1 + noise_kp1)

        # force update
        self._model.apply_boundary_conditions()
        self._model.update_force()

        # update the chaced noise
        self._noise_k1 = noise_kp1

# The Euler-Maruyama memthod
class EulerMaruyama(OverDampedLangevinIntegrator):
    """ Class which implements the Euler-Maruyama method
    which is a first order dynamics integrator
    """
    
    def __init__(self,model, stepsize, inverse_temperature):
        """ Init function for the class
        
        :param stepsize: discrete stepsize for time
        :param model: class which is used to update the force
        :param inverse_temperature: scaling for the gradient - beta
        """
        if model.p is not None:
            raise ValueError('EulerMaruyama is only first Order Dynamics!')
        
        super(EulerMaruyama, self).__init__(model, stepsize, inverse_temperature)

        self._zeta = np.sqrt(2. * self._stepsize / self._inverse_temperature)
        
    def integrate(self):
        """ Integration function which evolves the model
        given in self._model
        """
        
        # step method forward in time
        self._model.q += self._stepsize * self._model.f + self._zeta * np.random.normal(0., 1., self._model._dim)

        # force update
        self._model.apply_boundary_conditions()
        self._model.update_force()

# Heun's method
class HeunsMethod(OverDampedLangevinIntegrator):
    """ Class which implements Heun's method
    which is a first order dynamics integrator
    """
    def __init__(self, model, stepsize, inverse_temperature):
        """ Init function for the class
        
        :param stepsize: discrete stepsize for time
        :param model: class which is used to update the force
        :param inverse_temperature: scaling for the gradient - beta
        """
        if model.p is not None:
            raise ValueError('HeunsMethod is only first Order Dynamics!')
        
        super(HeunsMethod, self).__init__(model, stepsize, inverse_temperature)

        self._zeta = np.sqrt(2. * self._stepsize / self._inverse_temperature)
    
    def integrate(self):
        """ Integration function which evolves the model
        given in self._model
        """
        # cache some data 
        noise_cache = np.random.normal(0., 1., self._model._dim)

        # preforce update
        q_cache = self._model.q + self._stepsize * self._model.f + self._zeta * noise_cache

        # force update #1
        force_cache = self._model.get_force(q_cache)
        
        # post intermediate force update
        self._model.q += .5 * self._stepsize * (force_cache + self._model.f) + self._zeta * noise_cache
        
        # force update #2
        self._model.apply_boundary_conditions()
        self._model.update_force()

# Second Order Dynamics Langevin Integrators
class LangevinIntegrator(Sampler):
    """ Base class for all Langevin based integrators
    """
    
    def __init__(self, model, stepsize, inverse_temperature, friction_constant):
        """ Init function for the class

        :param stepsize: discrete stepsize for time
        :param model: class which is used to update the force
        :param inverse_temperature: scaling for the gradient - beta
        :param friction_constant: constant which scales the momentum - gamma
        """
        super(LangevinIntegrator, self).__init__(model, stepsize)
        
        self._inverse_temperature = inverse_temperature
        self._friction_constant = friction_constant

        self._alpha = np.exp(-self._friction_constant * self._stepsize)
        self._zeta = np.sqrt(1.0 - self._alpha * self._alpha)
        
    def integrate(self):
        """ Function which implements the integration forward in time
        for the molecule.

        :param molecule: object of type molecule which holds data
        """
        raise NotImplementedError()

    def A(self,p,scale):
        """ Function that performs the common A
        step, returns q
        
        :param p: momentum
        :param scale: scaling of the stepsize 
        """
        return scale * self._stepsize * p
    
    def B(self,f,scale):
        """ Function that performs the common B
        step, returns p
        
        :param f: force
        :param scale: scaling of the stepsize 
        """
        return scale * self._stepsize * f
    
class BAOAB(LangevinIntegrator):   
    """ The BAOAB Langevin Integrator class
    """
    
    def integrate(self):
        """ Function which implements the integration forward in time
        for the molecule.
        :param molecule: object of type molecule which holds data
        """
        # pre force integration steps
        self._model.p += self.B(self._model.f,.5)
        self._model.q += self.A(self._model.p,.5)
        self._model.p = self._alpha * self._model.p + self._zeta * np.random.normal(0., 1., self._model._dim)
        self._model.q += self.A(self._model.p,.5)

        # force update
        self._model.apply_boundary_conditions()
        self._model.update_force()

        # post-force integration steps
        self._model.p += self.B(self._model.f,.5)

class EnsembleQuasiNewton(LangevinIntegrator):
    """ Base class for samplers implementing multiple replicas of the target system
    """
    
    def __init__(self, repmodel, stepsize, inverse_temperature, friction_constant,regparams=1.0, B_update_mod=1):

        if not isinstance(repmodel, models.ReplicatedModel):
            ValueError("repmodel needs to be an instance of class 'ReplicatedModel'")
        
        super(EnsembleQuasiNewton, self).__init__(repmodel, stepsize, inverse_temperature, friction_constant)
        self.Bmatrix = np.zeros([self._model._nreplicas, self._model._model._dim,self._model._model._dim])
        for i in range(self._model._nreplicas):
            self.Bmatrix[i,:,:] = np.eye( self._model._model._dim)
        self._regparams = regparams
        self._B_update_mod = B_update_mod
        self._substep_counter = 0
        
    

    def integrate(self):
        
        nreplicas = self._model._nreplicas
           
         # update preconditioner
        if self._substep_counter % self._B_update_mod == 0:
            self.update_Bmatrix()
            
        # B-step
        for i in range(nreplicas):
            self._model.q[i,:] += .5 * self._stepsize  * np.matmul(self.Bmatrix[i,:,:], self._model.p[i,:].flatten())
            
        # A-step
        for i in range(nreplicas):
            self._model.p[i,:] += .5 * self._stepsize * np.matmul(np.transpose(self.Bmatrix[i,:,:]), self._model.f[i,:].flatten())
        
        # O-step
        self._model.p = self._alpha * self._model.p + self._zeta * np.random.normal(0., 1., [nreplicas,self._model._model._dim])
         
        # A-step   
        for i in range(nreplicas):
            self._model.p[i,:] += .5 * self._stepsize * np.matmul(np.transpose(self.Bmatrix[i,:,:]), self._model.f[i,:].flatten())
        
        # update force
        self._model.apply_boundary_conditions()
        self._model.update_force()
        
        # B-step
        for i in range(nreplicas):
            self._model.q[i,:] += .5 * self._stepsize * np.matmul(self.Bmatrix[i,:,:], self._model.p[i,:].flatten()) 
         
       
            
        self._substep_counter+=1
        
    def update_Bmatrix(self):
        if self._model._nreplicas > 1:
            indices = [i for i in range(self._model._nreplicas)]
            for r in range(self._model._nreplicas):
                mask =  np.array(indices[:r] + indices[(r + 1):])
                self.Bmatrix[r,:,:] = np.linalg.cholesky(
                        np.cov(self._model.q[mask,:],rowvar=False) + self._regparams * np.eye(self._model._model._dim)
                                                        )


def autocorr(x, maxlag=100):
    acf_vec = np.zeros(maxlag)
    xmean = np.mean(x)
    n = x.shape[0]
    for lag in range(maxlag):
        index = np.arange(0,n-lag,1)
        index_shifted = np.arange(lag,n,1)
        acf_vec[lag] = np.mean((x[index ]-xmean)*(x[index_shifted]-xmean))
    
    return acf_vec   
            