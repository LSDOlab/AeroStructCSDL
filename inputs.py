import csdl
import python_csdl_backend
import numpy as np

class inputs(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('num_variables',default=18)
        self.parameters.declare('bc')
        self.parameters.declare('seq')
    def define(self):
        n = self.parameters['num_nodes']
        num_variables = self.parameters['num_variables']
        bc = self.parameters['bc']
        seq = self.parameters['seq']


        E = self.create_output('E',shape=(3,3,n),val=0)
        for i in range(n):
            E[0,0,i] = EIxx[i]
            # E[i][0, 1] = 0
            E[0,2,i] = EIxz[i]
            # E[i][1, 0] = 0
            E[1,1,i] = GJ[i]
            # E[i][1, 2] = 0
            E[2,0,i] = EIxz[i]
            # E[i][2, 1] = 0
            E[2,2,i] = EIzz[i]