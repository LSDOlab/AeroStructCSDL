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
        uDot = self.declare_variable('uDot',shape=(3,n),val=0)
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
            
            # for OMEGA X ui
            uiT[i,0, 0] = csdl.expand(zero,(1,1,1))
            uiT[i,0, 1] = csdl.expand(-ri[2,0],(1,1,1),'ij->ijk')
            uiT[i,0, 2] = csdl.expand(ri[1,0],(1,1,1),'ij->ijk')
            uiT[i,1, 0] = csdl.expand(ri[2,0],(1,1,1),'ij->ijk')
            uiT[i,1, 1] = csdl.expand(zero,(1,1,1))
            uiT[i,1, 2] = csdl.expand(-ri[0,0],(1,1,1),'ij->ijk')
            uiT[i,2, 0] = csdl.expand(-ri[1,0],(1,1,1),'ij->ijk')
            uiT[i,2, 1] = csdl.expand(ri[0,0],(1,1,1),'ij->ijk')
            uiT[i,2, 2] = csdl.expand(zero,(1,1,1))
            
            # a_i (UNS, Eq. 23, Page 7)
            # a_i[:, i] = A0 + uDoti + mtimes(riT, ALPHA0) + mtimes(inner3T, OMEGA) + 2 * mtimes(uiT, OMEGA)
            term_1 = csdl.expand(A0,(3,1),'i->ij') + uDoti # (3,1)
            term_2 = csdl.expand(csdl.matvec(collapsed_riT,ALPHA0),(3,1),'i->ij') # (3,1)
            collapsed_inner3T = csdl.reshape(inner3T[i,:,:],new_shape=(3,3)) # (3,3)
            term_3 = csdl.expand(csdl.matvec(collapsed_inner3T,OMEGA),(3,1),'i->ij') # (3,1)
            collapsed_uiT = csdl.reshape(uiT[i,:,:],new_shape=(3,3)) # (3,3)
            term_4 = 2*csdl.expand(csdl.matvec(collapsed_uiT,OMEGA),(3,1),'i->ij') # (3,1)
            a_i[:,i] = term_1 + term_2 + term_3 + term_4

        


        delta_rCG_tilde = self.declare_variable('delta_rCG_tilde',shape=(3,3,n-1))
        omega = self.declare_variable('omega',shape=(3,n))
        omegaDot = self.declare_variable('omegaDot',shape=(3,n),val=0)
        innerT = self.create_output('innerT',shape=(n-1,3,3))
        inner2T = self.create_output('inner2T',shape=(n-1,3,3))

        # acceleration of the element
        for i in range(0, n - 1):
            # current element quantities
            drCG = delta_rCG_tilde[:,:,i]
            collapsed_drCG = csdl.reshape(drCG,new_shape=(3,3))
            
            # current node quantities
            omi = omega[:, i]
            omDoti = omegaDot[:, i]
            ai = a_i[:, i]
            # next node quantities
            omi1 = omega[:, i + 1]
            omDoti1 = omegaDot[:, i + 1]
            ai1 = a_i[:, i + 1]
            
            # for OMEGA X (OMEGA X delta_rCG)
            inner = csdl.matvec(collapsed_drCG,OMEGA)
            innerT[i,0, 0] = csdl.expand(zero,(1,1,1))
            innerT[i,0, 1] = csdl.expand(-inner[2],(1,1,1))
            innerT[i,0, 2] = csdl.expand(inner[1],(1,1,1))
            innerT[i,1, 0] = csdl.expand(inner[2],(1,1,1))
            innerT[i,1, 1] = csdl.expand(zero,(1,1,1))
            innerT[i,1, 2] = csdl.expand(-inner[0],(1,1,1))
            innerT[i,2, 0] = csdl.expand(-inner[1],(1,1,1))
            innerT[i,2, 1] = csdl.expand(inner[0],(1,1,1))
            innerT[i,2, 2] = csdl.expand(zero,(1,1,1))
            
            # for omega_i X (omega_i X delta_rCG)
            vec = csdl.reshape((0.5 * (omi + omi1)),new_shape=(3,))
            inner2 = csdl.matvec(collapsed_drCG,vec)
            
            inner2T[i,0, 0] = csdl.expand(zero,(1,1,1))
            inner2T[i,0, 1] = csdl.expand(-inner2[2],(1,1,1))
            inner2T[i,0, 2] = csdl.expand(inner2[1],(1,1,1))
            inner2T[i,1, 0] = csdl.expand(inner2[2],(1,1,1))
            inner2T[i,1, 1] = csdl.expand(zero,(1,1,1))
            inner2T[i,1, 2] = csdl.expand(-inner2[0],(1,1,1))
            inner2T[i,2, 0] = csdl.expand(-inner2[1],(1,1,1))
            inner2T[i,2, 1] = csdl.expand(inner2[0],(1,1,1))
            inner2T[i,2, 2] = csdl.expand(zero,(1,1,1))
            
            # nodal a_cg (UNS, eq. 38, Page 9)
            term_1 = 0.5 * (ai + ai1)
            term_2 = csdl.expand(csdl.matvec(collapsed_drCG,(ALPHA0 + csdl.reshape((0.5 * (omDoti + omDoti1)),new_shape=(3,)))), (3,1),'i->ij')
            collapsed_innerT = csdl.reshape(innerT[i,:,:],new_shape=(3,3))
            term_3 = csdl.expand(csdl.matvec(collapsed_innerT,OMEGA), (3,1),'i->ij')
            collapsed_inner2T = csdl.reshape(inner2T[i,:,:],new_shape=(3,3))
            term_4 = csdl.expand(csdl.matvec(collapsed_inner2T,csdl.reshape((0.5 * (omi + omi1)),new_shape=(3,))), (3,1),'i->ij')
            term_5 = csdl.expand(2*csdl.matvec(collapsed_inner2T,OMEGA), (3,1),'i->ij')

            aCG[:, i] = term_1 + term_2 + term_3 + term_4 + term_5