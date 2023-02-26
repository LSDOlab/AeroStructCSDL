import csdl
import python_csdl_backend
import numpy as np


class CalcNodalK(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('seq')
    def define(self):
        n = self.parameters['num_nodes']
        seq = self.parameters['seq']

        th = self.declare_variable('theta',shape=(3,n))

        K = self.create_output('K',shape=(3,3,n))
        Ka = self.create_output('Ka',shape=(3,3,n-1))

        one = self.declare_variable('one',val=1)
        zero = self.declare_variable('zero',val=0)

        for i in range(0, n):

            # surface beam
            if not (seq - [1, 3, 2]).any():
                K[0,0,i] = csdl.cos(th[2, i]) * csdl.cos(th[1, i])
                K[0,1,i] = 0
                K[0,2,i] = -csdl.sin(th[1, i])
                K[1,0,i] = -csdl.sin(th[2, i])
                K[1,1,i] = 1
                K[1,2,i] = 0
                K[2,0,i] = csdl.cos(th[2, i]) * csdl.sin(th[1, i])
                K[2,1,i] = 0
                K[2,2,i] = csdl.cos(th[1, i])

            
            # fuselage beam
            if not (seq - [3, 1, 2]).any():
                K[0,0,i] = csdl.expand(csdl.cos(th[1, i]),(1,1,1),'ij->ijk')
                K[0,1,i] = csdl.expand(zero,(1,1,1))
                K[0,2,i] = csdl.expand(-csdl.cos(th[0, i]) * csdl.sin(th[1, i]),(1,1,1),'ij->ijk')
                K[1,0,i] = csdl.expand(zero,(1,1,1))
                K[1,1,i] = csdl.expand(one,(1,1,1))
                K[1,2,i] = csdl.expand(csdl.sin(th[0, i]),(1,1,1),'ij->ijk')
                K[2,0,i] = csdl.expand(csdl.sin(th[1, i]),(1,1,1),'ij->ijk')
                K[2,1,i] = csdl.expand(zero,(1,1,1))
                K[2,2,i] = csdl.expand(csdl.cos(th[0, i]) * csdl.cos(th[1, i]),(1,1,1),'ij->ijk')

            
            if i >= 1:
                Ka[:,:,i-1] = (K[:,:,i-1] + K[:,:,i]) / 2
            