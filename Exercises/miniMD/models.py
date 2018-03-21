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
        """ updates the force internally from the current position
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

        
