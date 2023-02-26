import csdl
import python_csdl_backend
import numpy as np


class CalcNodalT(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('seq')
    def define(self):
        n = self.parameters['num_nodes']
        seq = self.parameters['seq']

        T = self.create_output('T',shape=(3,3,n))
        Ta = self.create_output('Ta',shape=(3,3,n-1))
        R = self.create_output('R', shape=(n,3,3,3))

        th = self.declare_variable('theta',shape=(3,n))

        one = self.declare_variable('one',val=1)
        zero = self.declare_variable('zero',val=0)

        for i in range(0, n):
            a1 = th[0, i]
            a2 = th[1, i]
            a3 = th[2, i]

            # rotation tensor for "phi" rotation (angle a1)
            R[i,0,0,0] = csdl.expand(one,(1,1,1,1))
            R[i,0,0,1] = csdl.expand(zero,(1,1,1,1))
            R[i,0,0,2] = csdl.expand(zero,(1,1,1,1))
            R[i,0,1,0] = csdl.expand(zero,(1,1,1,1))
            R[i,0,1,1] = csdl.expand(csdl.cos(a1),(1,1,1,1),'ij->ijkl')
            R[i,0,1,2] = csdl.expand(csdl.sin(a1),(1,1,1,1),'ij->ijkl')
            R[i,0,2,0] = csdl.expand(zero,(1,1,1,1))
            R[i,0,2,1] = csdl.expand(-csdl.sin(a1),(1,1,1,1),'ij->ijkl')
            R[i,0,2,2] = csdl.expand(csdl.cos(a1),(1,1,1,1),'ij->ijkl')

            # rotation tensor for "theta" rotation (angle a2)
            R[i,1,0,0] = csdl.expand(csdl.cos(a2),(1,1,1,1),'ij->ijkl')
            R[i,1,0,1] = csdl.expand(zero,(1,1,1,1))
            R[i,1,0,2] = csdl.expand(-csdl.sin(a2),(1,1,1,1),'ij->ijkl')
            R[i,1,1,0] = csdl.expand(zero,(1,1,1,1))
            R[i,1,1,1] = csdl.expand(one,(1,1,1,1))
            R[i,1,1,2] = csdl.expand(zero,(1,1,1,1))
            R[i,1,2,0] = csdl.expand(csdl.sin(a2),(1,1,1,1),'ij->ijkl')
            R[i,1,2,1] = csdl.expand(zero,(1,1,1,1))
            R[i,1,2,2] = csdl.expand(csdl.cos(a2),(1,1,1,1),'ij->ijkl')

            # rotation tensor for "psi" rotation (angle a3)
            R[i,2,0,0] = csdl.expand(csdl.cos(a3),(1,1,1,1),'ij->ijkl')
            R[i,2,0,1] = csdl.expand(csdl.sin(a3),(1,1,1,1),'ij->ijkl')
            R[i,2,0,2] = csdl.expand(zero,(1,1,1,1))
            R[i,2,1,0] = csdl.expand(-csdl.sin(a3),(1,1,1,1),'ij->ijkl')
            R[i,2,1,1] = csdl.expand(csdl.cos(a3),(1,1,1,1),'ij->ijkl')
            R[i,2,1,2] = csdl.expand(zero,(1,1,1,1))
            R[i,2,2,0] = csdl.expand(zero,(1,1,1,1))
            R[i,2,2,1] = csdl.expand(zero,(1,1,1,1))
            R[i,2,2,2] = csdl.expand(one,(1,1,1,1))

            # multiply single-axis rotation tensors in reverse order
            # T[i][:, :] = mtimes(R[seq[2] - 1][:, :], mtimes(R[seq[1] - 1][:, :], R[seq[0] - 1][:, :]))

            term_1 = csdl.reshape(R[i,(int(seq[1])-1),:,:],new_shape=(3,3))
            term_2 = csdl.reshape(R[i,(int(seq[0])-1),:,:],new_shape=(3,3))
            inner_term = csdl.matmat(term_1, term_2)
            
            T[:,:,i] = csdl.expand(csdl.matmat(csdl.reshape(R[i,(int(seq[2])-1),:,:],new_shape=(3,3)), inner_term), (3,3,1),'ij->ijk')

            
            if i >= 1:
                Ta[:,:,i-1] = (T[:,:,i-1] + T[:,:,i]) / 2
            