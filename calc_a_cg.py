import csdl
import python_csdl_backend
import numpy as np


class calc_a_cg(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
    def define(self):
        n = self.parameters['num_nodes']

        r = self.declare_variable('r',shape=(3,n))
        u = self.declare_variable('u',shape=(3,n))
        uDot = self.declare_variable('uDot',shape=(3,n))
        A0 = self.declare_variable('A0',shape=(3))
        OMEGA = self.declare_variable('OMEGA',shape=(3))
        ALPHA0 = self.declare_variable('ALPHA0',shape=(3))

        aCG = self.create_output('aCG',shape=(3,n-1))
        a_i = self.create_output('a_i',shape=(3,n))
        riT = self.create_output('riT',shape=(n,3,3))
        uiT = self.create_output('uiT',shape=(n,3,3))
        inner3T = self.create_output('inner3T',shape=(n,3,3))

        zero = self.declare_variable('zero',val=0)

        for i in range(0, n):
            # current node quantities:
            ri = r[:, i]
            ui = u[:, i]
            uDoti = uDot[:, i]
            # riT = SX.sym('riT', 3, 3)
            # uiT = SX.sym('uiT', 3, 3)
            # inner3T = SX.sym('inner3T', 3, 3)
            # for OMEGA X (OMEGA X ri):
            riT[i,0, 0] = csdl.expand(zero,(1,1,1))
            riT[i,0, 1] = csdl.expand(-ri[2,0],(1,1,1),'ij->ijk')
            riT[i,0, 2] = csdl.expand(ri[1,0],(1,1,1),'ij->ijk')
            riT[i,1, 0] = csdl.expand(ri[2,0],(1,1,1),'ij->ijk')
            riT[i,1, 1] = csdl.expand(zero,(1,1,1))
            riT[i,1, 2] = csdl.expand(-ri[0,0],(1,1,1),'ij->ijk')
            riT[i,2, 0] = csdl.expand(-ri[1,0],(1,1,1),'ij->ijk')
            riT[i,2, 1] = csdl.expand(ri[0,0],(1,1,1),'ij->ijk')
            riT[i,2, 2] = csdl.expand(zero,(1,1,1))

            # inner3 = mtimes(riT, OMEGA)
            collapsed_riT = csdl.reshape(riT[i,:,:],new_shape=(3,3))
            # inner3 = csdl.matvec(riT[i,:,:],OMEGA)
            inner3 = csdl.matvec(collapsed_riT,OMEGA)

            
            inner3T[i,0, 0] = csdl.expand(zero,(1,1,1))
            inner3T[i,0, 1] = csdl.expand(-inner3[2],(1,1,1))
            inner3T[i,0, 2] = csdl.expand(inner3[1],(1,1,1))
            inner3T[i,1, 0] = csdl.expand(inner3[2],(1,1,1))
            inner3T[i,1, 1] = csdl.expand(zero,(1,1,1))
            inner3T[i,1, 2] = csdl.expand(-inner3[0],(1,1,1))
            inner3T[i,2, 0] = csdl.expand(-inner3[1],(1,1,1))
            inner3T[i,2, 1] = csdl.expand(inner3[0],(1,1,1))
            inner3T[i,2, 2] = csdl.expand(zero,(1,1,1))
            """
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



        # acceleration of the element
        for i in range(0, n - 1):
            innerT = SX.sym('innerT', 3, 3)
            inner2T = SX.sym('inner3T', 3, 3)
            # current element quantities
            drCG = delta_rCG_tilde[i][:, :]
            # current node quantities
            omi = omega[:, i]
            omDoti = omegaDot[:, i]
            ai = a_i[:, i]
            # next node quantities
            omi1 = omega[:, i + 1];
            omDoti1 = omegaDot[:, i + 1]
            ai1 = a_i[:, i + 1]

            # for OMEGA X (OMEGA X delta_rCG)
            inner = mtimes(drCG, OMEGA)
            innerT[0, 0] = 0
            innerT[0, 1] = -inner[2]
            innerT[0, 2] = inner[1]
            innerT[1, 0] = inner[2]
            innerT[1, 1] = 0
            innerT[1, 2] = -inner[0]
            innerT[2, 0] = -inner[1]
            innerT[2, 1] = inner[0]
            innerT[2, 2] = 0
            # for omega_i X (omega_i X delta_rCG)
            inner2 = mtimes(drCG, (0.5 * (omi + omi1)))
            inner2T[0, 0] = 0
            inner2T[0, 1] = -inner2[2]
            inner2T[0, 2] = inner2[1]
            inner2T[1, 0] = inner2[2]
            inner2T[1, 1] = 0
            inner2T[1, 2] = -inner2[0]
            inner2T[2, 0] = -inner2[1]
            inner2T[2, 1] = inner2[0]
            inner2T[2, 2] = 0
            # nodal a_cg (UNS, eq. 38, Page 9)
            aCG[:, i] = 0.5 * (ai + ai1) + mtimes(drCG, (ALPHA0 + (0.5 * (omDoti + omDoti1)))) + mtimes(innerT,
                                                                                                        OMEGA) + mtimes(
                inner2T, (
                        0.5 * (omi + omi1))) + 2 * mtimes(inner2T, OMEGA)
        # return aCG
        """