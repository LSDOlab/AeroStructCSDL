import csdl
import python_csdl_backend
import numpy as np


class calc_a_cg(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
    def define(self):
        n = self.parameters['num_nodes']

        r = self.declare_variable('r',shape=(n))
        u = self.declare_variable('u',shape=(n))
        uDot = self.declare_variable('uDot',shape=(n))
        A0 = self.declare_variable('A0',shape=(n)) # shape ?
        OMEGA = self.declare_variable('OMEGA',shape=(n)) # shape ?
        ALPHA0 = self.declare_variable('ALPHA0',shape=(n)) # shape ?

        aCG = self.create_output('aCG',shape=(3,n-1))
        a_i = self.create_output('a_i',shape=(3,n))
        riT = self.create_output('riT',shape=(n,3,3))
        uiT = self.create_output('uiT',shape=(n,3,3))
        inner3T = self.create_output('inner3T',shape=(n,3,3))

        for i in range(0, n):
            # current node quantities:
            ri = r[:, i]
            ui = u[:, i]
            uDoti = uDot[:, i]
            # riT = SX.sym('riT', 3, 3)
            # uiT = SX.sym('uiT', 3, 3)
            # inner3T = SX.sym('inner3T', 3, 3)
            # for OMEGA X (OMEGA X ri):
            riT[i,0, 0] = 0
            riT[i,0, 1] = -ri[2]
            riT[i,0, 2] = ri[1]
            riT[i,1, 0] = ri[2]
            riT[i,1, 1] = 0
            riT[i,1, 2] = -ri[0]
            riT[i,2, 0] = -ri[1]
            riT[i,2, 1] = ri[0]
            riT[i,2, 2] = 0

            #inner3 = mtimes(riT, OMEGA)
            inner3 = csdl.matmat(riT[i,:,:],OMEGA[i,:,:])

            inner3T[i,0, 0] = 0
            inner3T[i,0, 1] = -inner3[2]
            inner3T[i,0, 2] = inner3[1]
            inner3T[i,1, 0] = inner3[2]
            inner3T[i,1, 1] = 0
            inner3T[i,1, 2] = -inner3[0]
            inner3T[i,2, 0] = -inner3[1]
            inner3T[i,2, 1] = inner3[0]
            inner3T[i,2, 2] = 0

            # for OMEGA X ui
            uiT[i,0, 0] = 0
            uiT[i,0, 1] = -ri[2]
            uiT[i,0, 2] = ri[1]
            uiT[i,1, 0] = ri[2]
            uiT[i,1, 1] = 0
            uiT[i,1, 2] = -ri[0]
            uiT[i,2, 0] = -ri[1]
            uiT[i,2, 1] = ri[0]
            uiT[i,2, 2] = 0

            # a_i (UNS, Eq. 23, Page 7)
            #a_i[:, i] = A0 + uDoti + mtimes(riT, ALPHA0) + mtimes(inner3T, OMEGA) + 2 * mtimes(uiT, OMEGA)
            a_i[:,i] = A0 + uDoti + csdl.matmat(riT[i,:,:],ALPHA0) + csdl.matmat(inner3T[i,:,:],OMEGA) + 2*csdl.matmat(uiT[i,:,:],OMEGA)

