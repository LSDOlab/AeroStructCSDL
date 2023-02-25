import csdl
import python_csdl_backend
import numpy as np



class calcT_ac(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
    def define(self):
        n = self.parameters['num_nodes']

        TH = self.declare_variable('TH',shape=(3)) # shape ???????????

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

        # Calc 3 matrices
        R_psi[0, 0] = csdl.cos(PSI)
        R_psi[0, 1] = csdl.sin(PSI)
        R_psi[0, 2] = zero
        R_psi[1, 0] = -csdl.sin(PSI)
        R_psi[1, 1] = csdl.cos(PSI)
        R_psi[1, 2] = zero
        R_psi[2, 0] = zero
        R_psi[2, 1] = zero
        R_psi[2, 2] = one

        R_th[0, 0] = csdl.cos(THETA)
        R_th[0, 1] = zero
        R_th[0, 2] = csdl.sin(THETA)
        R_th[1, 0] = zero
        R_th[1, 1] = one
        R_th[1, 2] = zero
        R_th[2, 0] = -csdl.sin(THETA)
        R_th[2, 1] = zero
        R_th[2, 2] = csdl.cos(THETA)

        R_phi[0, 0] = one
        R_phi[0, 1] = zero
        R_phi[0, 2] = zero
        R_phi[1, 0] = zero
        R_phi[1, 1] = csdl.cos(PHI)
        R_phi[1, 2] = csdl.sin(PHI)
        R_phi[2, 0] = zero
        R_phi[2, 1] = -csdl.sin(PHI)
        R_phi[2, 2] = csdl.cos(PHI)

        # calculate rotation matrix
        # T_E = mtimes(R_psi, mtimes(R_th, R_phi))
        T_E = csdl.matmat(R_psi, csdl.matmat(R_th,R_phi))
        self.register_output('T_E',T_E)