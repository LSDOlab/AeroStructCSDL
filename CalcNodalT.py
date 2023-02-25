import csdl
import python_csdl_backend
import numpy as np


class CalcNodalT(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
    def define(self):
        n = self.parameters['num_nodes']

        T = self.create_output('T',shape=(3,3,n))
        Ta = self.create_output('Ta',shape=(3,3,n-1))
        R = self.create_output('R', shape=(3,3,3))

        th = self.declare_variable('th',shape=(n)) # shape ?
        seq = self.declare_variable('seq',shape=(n)) # shape ?

        for i in range(0, n):
            a1 = th[0, i]
            a2 = th[1, i]
            a3 = th[2, i]
            # rotation tensor for "phi" rotation (angle a1)
            R[0][0, 0] = 1
            R[0][0, 1] = 0
            R[0][0, 2] = 0
            R[0][1, 0] = 0
            R[0][1, 1] = csdl.cos(a1)
            R[0][1, 2] = csdl.sin(a1)
            R[0][2, 0] = 0
            R[0][2, 1] = -csdl.sin(a1)
            R[0][2, 2] = csdl.cos(a1)
            # rotation tensor for "theta" rotation (angle a2)
            R[1][0, 0] = csdl.cos(a2)
            R[1][0, 1] = 0
            R[1][0, 2] = -csdl.sin(a2)
            R[1][1, 0] = 0
            R[1][1, 1] = 1
            R[1][1, 2] = 0
            R[1][2, 0] = csdl.sin(a2)
            R[1][2, 1] = 0
            R[1][2, 2] = csdl.cos(a2)
            # rotation tensor for "psi" rotation (angle a3)
            R[2][0, 0] = csdl.cos(a3)
            R[2][0, 1] = csdl.sin(a3)
            R[2][0, 2] = 0
            R[2][1, 0] = -csdl.sin(a3)
            R[2][1, 1] = csdl.cos(a3)
            R[2][1, 2] = 0
            R[2][2, 0] = 0
            R[2][2, 1] = 0
            R[2][2, 2] = 1
            # multiply single-axis rotation tensors in reverse order
            T[i][:, :] = mtimes(R[seq[2] - 1][:, :], mtimes(R[seq[1] - 1][:, :], R[seq[0] - 1][:, :]))


            if i >= 1: # ??????????????????????????
                Ta[i - 1][:, :] = (T[i - 1][:, :] + T[i][:, :]) / 2