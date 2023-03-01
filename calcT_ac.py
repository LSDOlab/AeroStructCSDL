import csdl
import python_csdl_backend
import numpy as np



class calcT_ac(csdl.Model):
    def initialize(self):
        pass
    def define(self):
        TH = self.declare_variable('THETA',shape=(3))

        # rotation matrix (3x3) that rotates a vector in xyz to XYZ as follows:
        # A_XYZ = T_E * A_xyz
        R_psi = self.create_output('R_psi',shape=(3,3))
        R_th = self.create_output('R_th',shape=(3,3))
        R_phi = self.create_output('R_phi',shape=(3,3))
        # read aircraft states (rad)
        PHI = TH[0]
        THETA = TH[1]
        PSI = TH[2]

        # dummy vars
        one = self.create_input('one',val=1.0)
        zero = self.create_input('zero',val=0.0)
        
        # calc 3 matrices
        R_psi[0, 0] = csdl.expand(csdl.cos(PSI),(1,1))
        R_psi[0, 1] = csdl.expand(csdl.sin(PSI),(1,1))
        R_psi[0, 2] = csdl.expand(zero,(1,1))
        R_psi[1, 0] = csdl.expand(-csdl.sin(PSI),(1,1))
        R_psi[1, 1] = csdl.expand(csdl.cos(PSI),(1,1))
        R_psi[1, 2] = csdl.expand(zero,(1,1))
        R_psi[2, 0] = csdl.expand(zero,(1,1))
        R_psi[2, 1] = csdl.expand(zero,(1,1))
        R_psi[2, 2] = csdl.expand(one,(1,1))
        
        R_th[0, 0] = csdl.expand(csdl.cos(THETA),(1,1))
        R_th[0, 1] = csdl.expand(zero,(1,1))
        R_th[0, 2] = csdl.expand(csdl.sin(THETA),(1,1))
        R_th[1, 0] = csdl.expand(zero,(1,1))
        R_th[1, 1] = csdl.expand(one,(1,1))
        R_th[1, 2] = csdl.expand(zero,(1,1))
        R_th[2, 0] = csdl.expand(-csdl.sin(THETA),(1,1))
        R_th[2, 1] = csdl.expand(zero,(1,1))
        R_th[2, 2] = csdl.expand(csdl.cos(THETA),(1,1))
        
        R_phi[0, 0] = csdl.expand(one,(1,1))
        R_phi[0, 1] = csdl.expand(zero,(1,1))
        R_phi[0, 2] = csdl.expand(zero,(1,1))
        R_phi[1, 0] = csdl.expand(zero,(1,1))
        R_phi[1, 1] = csdl.expand(csdl.cos(PHI),(1,1))
        R_phi[1, 2] = csdl.expand(csdl.sin(PHI),(1,1))
        R_phi[2, 0] = csdl.expand(zero,(1,1))
        R_phi[2, 1] = csdl.expand(-csdl.sin(PHI),(1,1))
        R_phi[2, 2] = csdl.expand(csdl.cos(PHI),(1,1))
        
        # calculate rotation matrix
        # T_E = mtimes(R_psi, mtimes(R_th, R_phi))
        inner_term = csdl.matmat(R_th,R_phi)
        T_E = csdl.matmat(R_psi, inner_term)
        self.register_output('T_E',T_E)
        