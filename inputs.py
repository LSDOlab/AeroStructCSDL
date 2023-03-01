import csdl
import python_csdl_backend
import numpy as np
from BoxBeamRep import *



class inputs(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes',default=16)
        self.parameters.declare('num_variables',default=18)
    def define(self):
        n = self.parameters['num_nodes']
        num_variables = self.parameters['num_variables']

        self.create_input('eye',shape=(3,3),val=np.eye(3))
        self.create_input('t_epsilon',shape=(1,1),val=0.03)
        self.create_input('t_gamma',shape=(1,1),val=0.03)
        self.create_input('xac',shape=(18),val=np.zeros(18))


        self.create_input('Einv',shape=(3,3,n), val=Einv)
        self.create_input('delta_r_CG_tilde',shape=(3,3,n-1),val=delta_r_CG_tilde)
        self.create_input('i_matrix',shape=(3,3,n-1),val=i_matrix)
        self.create_input('oneover',shape=(3,3,n),val=oneover)
        self.create_input('D',shape=(3,3,n),val=D)
        self.create_input('mu',shape=(n-1),val=mu)
        self.create_input('K0a',shape=(n-1,3,3),val=K0a)