import csdl
import python_csdl_backend
import numpy as np
from calc_a_cg import calc_a_cg
from CalcNodalT import CalcNodalT
from CalcNodalK import CalcNodalK
from calcT_ac import calcT_ac



class ResJac(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('num_variables')
        self.parameters.declare('bc')
        self.parameters.declare('g',val=9.81)
    def define(self):
        n = self.parameters['num_nodes']
        num_variables = self.parameters['num_variables']
        bc = self.parameters['bc'] # boundary conditions
        g = self.parameters['g'] # gravity

        x = self.declare_variable('x',shape=n) # node state vector
        xd = self.declare_variable('xd',shape=n) # derivatives of state vector
        xac = self.declare_variable('xac',shape=num_variables) # aircraft state vector

        Res = self.create_output('Res',shape=(num_variables,n))

        # read x
        r = x[0:3, :]
        theta = x[3:6, :]
        F = x[6:9, :]
        M = x[9:12, :]
        u = x[12:15, :]
        omega = x[15:18, :]

        # read xDot
        rDot = xd[0:3, :]
        thetaDot = xd[3:6, :]
        FDot = xd[6:9, :]
        MDot = xd[9:12, :]
        uDot = xd[12:15, :]
        omegaDot = xd[15:18, :]

        # read the aircraft states
        R = xac[0:3]
        U = xac[3:6]
        A0 = xac[6:9]
        THETA = xac[9:12]
        OMEGA = xac[12:15]
        ALPHA0 = xac[15:18]

        # forces and moments
        f_aero = self.declare_variable('f_aero',shape=n)
        m_aero = self.declare_variable('m_aero',shape=n)
        delta_Fapplied = self.declare_variable('delta_Fapplied',shape=n)
        delta_Mapplied = self.declare_variable('delta_Mapplied',shape=n)

        # Read Stick Model
        mu = self.declare_variable('mu',shape=n)  # 1xn vector of mass/length
        seq = self.declare_variable('seq',shape=n)
        theta0 = self.declare_variable('theta0',shape=n)
        K0a = self.declare_variable('K0a',shape=n)
        delta_s0 = self.declare_variable('delta_s0',shape=n)

        i_matrix = self.create_output('i_matrix',shape=(3,3,n-1))
        delta_rCG_tilde = self.create_output('delta_rCG_tilde',shape=(3,3,n-1))
        Einv = self.create_output('Einv',shape=(3,3,n))
        D = self.create_output('D',shape=(3,3,n))
        oneover = self.create_output('oneover',shape=(3,3,n))


        # do nodal quantities of symbolic pieces in 3D matrices:
        j = 0
        for i in range(1): # single beam at present
            Einv[j][:, :] = beam_list['Einv'][i][:, :]
            D[j][:, :] = beam_list['D'][i][:, :]
            oneover[j][:, :] = beam_list['oneover'][i][:, :]
            j = j + 1

        # do element quatities of symbolic pieces in 3D matrices:
        j = 0
        for i in range(beam_list['inter_node_lim'][element, 0],
                       beam_list['inter_node_lim'][element, 1]):
            delta_rCG_tilde[j][:, :] = beam_list['delta_r_CG_tilde'][j][:, :]
            i_matrix[j][:, :] = beam_list['i_matrix'][j][:, :]
            j = j + 1

        self.add(calc_a_cg(num_nodes=n),name='calc_a_cg')
        a_cg = self.declare_variable('a_cg',shape=(n)) # shape ??

        # get T and K matrices:
        #T, Ta = CalcNodalT(theta, seq, n=n)
        self.add(CalcNodalT(num_nodes=n),name='CalcNodalT')
        T = self.declare_variable('T',shape=(n)) # shape ??????????????
        Ta = self.declare_variable('Ta',shape=(n)) # shape ??????????????
        #K, Ka = self.CalcNodalK(theta, seq)
        self.add(CalcNodalK(num_nodes=n),name='CalcNodalK')
        K = self.declare_variable('K',shape=(n)) # shape ??????????????
        Ka = self.declare_variable('Ka',shape=(n)) # shape ??????????????


        # gravity in body fixed axes
        # T_E = self.calcT_ac(THETA)  # UNS, Eq. 6, Page 5
        self.add(calcT_ac(),name='calcT_ac')
        T_E = self.declare_variable('T_E',shape=(3,3)) # shape ?????????
        # g_xyz = mtimes(transpose(T_E), g)
        g_xyz = csdl.matmat(csdl.transpose(T_E),g)
        f_acc = self.create_output('f_acc',shape=(3,n-1))
        m_acc = self.create_output('m_acc',shape=(3,n-1))


        for ind in range(0, n - 1):
            f_acc[:, ind] = csdl.matmat(mu[ind], (g_xyz - a_cg[:, ind]))
            TiT = csdl.matmat((0.5 * (csdl.transpose(T[ind][:, :]) + csdl.transpose(T[ind + 1][:, :]))),
                         csdl.matmat(i_matrix[ind][:, :], (0.5 * (T[ind][:, :] + T[ind + 1][:, :]))))
            m_acc[:, ind] = csdl.matmat(delta_rCG_tilde[ind][:, :], f_acc[:, ind]) - csdl.matmat(TiT, ALPHA0 + (
                    0.5 * (omegaDot[:, ind] + omegaDot[:, ind + 1]))) - csdl.cross(
                (OMEGA + 0.5 * (omega[:, ind] + omega[:, ind + 1])),
                csdl.matmat(TiT, (OMEGA + 0.5 * (omega[:, ind] + omega[:, ind + 1]))))