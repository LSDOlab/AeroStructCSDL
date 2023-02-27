import csdl
import python_csdl_backend
import numpy as np
from calc_a_cg import calc_a_cg
from CalcNodalT import CalcNodalT
from CalcNodalK import CalcNodalK
from calcT_ac import calcT_ac
from inv import solver




class ResJac(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('num_variables',default=18)
        #self.parameters.declare('bc')
        self.parameters.declare('element',default=0)
        self.parameters.declare('g',default=np.array([0,0,9.81]))
        self.parameters.declare('options')
        self.parameters.declare('seq')
    def define(self):
        n = self.parameters['num_nodes']
        num_variables = self.parameters['num_variables']
        #bc = self.parameters['bc'] # boundary conditions
        element = self.parameters['element']
        gravity = self.parameters['g'] # gravity
        g = self.create_input('g',val=gravity,shape=(3))
        options = self.parameters['options'] # options dictionary
        seq = self.parameters['seq']

        x = self.declare_variable('x',shape=(num_variables,n)) # state vector
        xd = self.declare_variable('xd',shape=(num_variables,n)) # derivatives of state vector
        xac = self.declare_variable('xac',shape=(num_variables)) # aircraft state vector
        R_prec = self.declare_variable('R_prec',shape=(24))
        
        # read x
        r = x[0:3, :]
        self.register_output('r',r)
        theta = x[3:6, :]
        self.register_output('theta',theta)
        F = x[6:9, :]
        self.register_output('F',F)
        M = x[9:12, :]
        self.register_output('M',M)
        u = x[12:15, :]
        self.register_output('u',u)
        omega = x[15:18, :]
        self.register_output('omega',omega)

        # read xDot
        rDot = xd[0:3, :]
        self.register_output('rDot',rDot)
        thetaDot = xd[3:6, :]
        self.register_output('thetaDot',thetaDot)
        FDot = xd[6:9, :]
        self.register_output('FDot',FDot)
        MDot = xd[9:12, :]
        self.register_output('MDot',MDot)
        uDot = xd[12:15, :]
        self.register_output('uDot',uDot)
        omegaDot = xd[15:18, :]
        self.register_output('omegaDot',omegaDot)
        
        # read the aircraft states
        R = xac[0:3]
        # self.register_output('R',R) # duplicate R variable output?
        U = xac[3:6]
        self.register_output('U',U)
        A0 = xac[6:9]
        self.register_output('A0',A0)
        THETA = xac[9:12]
        self.register_output('THETA',THETA)
        OMEGA = xac[12:15]
        self.register_output('OMEGA',OMEGA)
        ALPHA0 = xac[15:18]
        self.register_output('ALPHA0',ALPHA0)
        
        # forces and moments
        f_aero = self.declare_variable('f_aero',shape=(3,n-1),val=0)
        m_aero = self.declare_variable('m_aero',shape=(3,n-1),val=0)
        delta_Fapplied = self.declare_variable('delta_Fapplied',shape=(3,n),val=0)
        delta_Mapplied = self.declare_variable('delta_Mapplied',shape=(3,n),val=0)
        
        # read the stick model
        mu = self.declare_variable('mu',shape=n)  # 1xn vector of mass/length
        theta0 = self.declare_variable('theta0',shape=n)
        K0a = self.declare_variable('K0a',shape=(n,3,3))
        delta_s0 = self.declare_variable('delta_s0',shape=n)
        
        #i_matrix = self.create_output('i_matrix',shape=(3,3,n-1))
        #delta_rCG_tilde = self.create_output('delta_rCG_tilde',shape=(3,3,n-1))
        #Einv = self.create_output('Einv',shape=(3,3,n))
        #D = self.create_output('D',shape=(3,3,n))
        #oneover = self.create_output('oneover',shape=(3,3,n))

        i_matrix = self.declare_variable('i_matrix',shape=(3,3,n-1))
        delta_rCG_tilde = self.declare_variable('delta_rCG_tilde',shape=(3,3,n-1))
        Einv = self.declare_variable('Einv',shape=(3,3,n))
        D = self.declare_variable('D',shape=(3,3,n))
        oneover = self.declare_variable('oneover',shape=(3,3,n))

        """
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
        """

        self.add(calc_a_cg(num_nodes=n),name='calc_a_cg')
        a_cg = self.declare_variable('aCG',shape=(3,n-1))
        
        # get T and K matrices:
        self.add(CalcNodalT(num_nodes=n,seq=seq),name='CalcNodalT')
        T = self.declare_variable('T',shape=(3,3,n))
        Ta = self.declare_variable('Ta',shape=(3,3,n-1))
        
        self.add(CalcNodalK(num_nodes=n,seq=seq),name='CalcNodalK')
        K = self.declare_variable('K',shape=(3,3,n))
        Ka = self.declare_variable('Ka',shape=(3,3,n-1))

        
        # gravity in body fixed axes
        self.add(calcT_ac(),name='calcT_ac') # UNS, Eq. 6, Page 5
        T_E = self.declare_variable('T_E',shape=(3,3))
        
        g_xyz = csdl.matvec(csdl.transpose(T_E),g)
        f_acc = self.create_output('f_acc',shape=(3,n-1))
        m_acc = self.create_output('m_acc',shape=(3,n-1))

        
        for ind in range(0, n - 1):
            inner_term = csdl.expand(g_xyz,(3,1),'i->ij') - a_cg[:, ind]
            f_acc[:, ind] = csdl.expand(csdl.matvec(inner_term,mu[ind]),(3,1),'i->ij')


            collapsed_t_ind = csdl.reshape(T[:,:,ind], new_shape=(3,3))
            collapsed_t_ind_1 = csdl.reshape(T[:,:,ind+1], new_shape=(3,3))
            TiT_t_1 = 0.5 * (csdl.transpose(collapsed_t_ind) + csdl.transpose(collapsed_t_ind_1))
            inner_term_1 = csdl.reshape(i_matrix[:,:,ind], new_shape=(3,3))
            inner_term_2 = csdl.reshape(0.5 * (T[:,:,ind] + T[:,:,ind+1]), new_shape=(3,3))
            TiT_t_2 = csdl.matmat(inner_term_1, inner_term_2)

            TiT = csdl.matmat(TiT_t_1, TiT_t_2)


            collapsed_delta_rCG_tilde = csdl.reshape(delta_rCG_tilde[:,:,ind], new_shape=(3,3))
            collapsed_f_acc = csdl.reshape(f_acc[:, ind], new_shape=(3))
            m_t_1 = csdl.matvec(collapsed_delta_rCG_tilde, collapsed_f_acc)

            collapsed_omegaDot_ind = csdl.reshape(omegaDot[:,ind], new_shape=(3))
            collapsed_omegaDot_ind_1 = csdl.reshape(omegaDot[:,ind+1], new_shape=(3))
            inner_mt2_term2 = ALPHA0 + (0.5*(collapsed_omegaDot_ind + collapsed_omegaDot_ind_1))
            m_t_2 = csdl.matvec(TiT, inner_mt2_term2)
            
            cross_t1 = OMEGA + 0.5*(csdl.reshape(omega[:,ind], new_shape=(3)) + csdl.reshape(omega[:,ind+1], new_shape=(3)))
            cross_t2 = csdl.matvec(TiT, (OMEGA + 0.5*(csdl.reshape(omega[:,ind], new_shape=(3)) + csdl.reshape(omega[:,ind+1], new_shape=(3)))))
            m_t_3 = csdl.cross(cross_t1, cross_t2, axis=0)

            m_acc[:, ind] = csdl.expand(m_t_1 - m_t_2 - m_t_3, (3,1),'i->ij')



        
        Mcsn = self.create_output('Mcsn',shape=(3,n))
        Fcsn = self.create_output('Fcsn',shape=(3,n))
        Mcsnp = self.create_output('Mcsnp',shape=(3,n))
        strainsCSN = self.create_output('strainsCSN',shape=(3,n))
        damp_MK = self.create_output('damp_MK',shape=(3,n))
        
        # compute K inverse for damp_MK computation
        self.add(solver(num_nodes=n))
        K_inv = self.declare_variable('K_inv',shape=(3,3,n))

        
        for ind in range(0, n):
            # transform xyz -> csn (ASW, Eq. 14, page 6);
            collapsed_T = csdl.reshape(T[:,:,ind],new_shape=(3,3))
            collapsed_M = csdl.reshape(M[:,ind],new_shape=(3))
            collapsed_F = csdl.reshape(F[:,ind],new_shape=(3))
            Mcsn[:,ind] = csdl.expand(csdl.matvec(collapsed_T, collapsed_M),(3,1),'i->ij')
            Fcsn[:,ind] = csdl.expand(csdl.matvec(collapsed_T, collapsed_F),(3,1),'i->ij')
            
            # get Mcsn_prime (ASW, Eq. 18, page 8)
            mcsnp_t2 = csdl.matvec(csdl.transpose(csdl.reshape(D[:,:,ind], new_shape=(3,3))), csdl.reshape(Fcsn[:,ind], new_shape=(3)))
            Mcsnp[:,ind] = Mcsn[:,ind] + csdl.expand(mcsnp_t2,(3,1),'i->ij')

            # get strains (ASW, Eq. 19, page 8)
            collapsed_oneover = csdl.reshape(oneover[:,:,ind],new_shape=(3,3))
            collapsed_Fcsn = csdl.reshape(Fcsn[:,ind], new_shape=(3))
            strainsCSN_t1 = csdl.matvec(collapsed_oneover, collapsed_Fcsn)

            collapsed_D = csdl.reshape(D[:,:,ind], new_shape=(3,3))
            collapsed_Einv = csdl.reshape(Einv[:,:,ind], new_shape=(3,3))
            collapsed_Mcsnp = csdl.reshape(Mcsnp[:,ind], new_shape=(3))
            strainsCSN_t2 = csdl.matvec(collapsed_D, csdl.matvec(collapsed_Einv, collapsed_Mcsnp))

            strainsCSN[:, ind] = csdl.expand(strainsCSN_t1 + strainsCSN_t2, (3,1),'i->ij')
            
            # get damping vector for moment-curvature relationship
            collapsed_K_inv = csdl.reshape(K_inv[:,:,ind], new_shape=(3,3))
            collapsed_omega = csdl.reshape(omega[:,ind], new_shape=(3))
            damp_MK_t2 = csdl.matvec(collapsed_T, collapsed_omega)

            damp_MK[:, ind] = csdl.expand(csdl.matvec(collapsed_K_inv, damp_MK_t2), (3,1),'i->ij')




        # convert options to csdl inputs for damp
        t_epsilon = self.create_input('t_epsilon',shape=(1,1),val=options['t_epsilon'])
        t_gamma = self.create_input('t_gamma',shape=(1,1),val=options['t_gamma'])
        damp = self.create_output('damp',shape=(3,3),val=np.zeros((3,3)))
        damp[0, 0] = 1*t_epsilon
        damp[1, 1] = 1*t_gamma
        damp[2, 2] = 1*t_epsilon
        
        # get total distributed force
        f = f_aero + f_acc
        m = m_aero + m_acc
        
        # get average nodal reaction forces
        Fav = self.create_output('Fav',shape=(3,n-1))

        for i in range(0, n - 1):
            Fav[:, i] = 0.5*(F[:, i] + F[:, i + 1])
        # endsection

        
        eps = 1e-19
        # get delta_s and delta_r
        delta_r = self.create_output('delta_r',shape=(3,n-1))
        delta_s = self.create_output('delta_s',shape=(n-1))
        for i in range(0,n-1):
            delta_r[:, i] = (r[:, i + 1] - r[:, i] + eps)  # added a non zero number to avoid the 1/sqrt(dx) singularity at the zero length nodes
            delta_s[i] = csdl.reshape(((delta_r[0, i])**2 + (delta_r[1, i])**2 + (delta_r[2, i])**2)**0.5, new_shape=(1))  # based on ASW, Eq. 49, Page 12


        
        # potential variables to be set as bc (can't create these vars in the for loop...)
        #varRoot = self.create_output('vr',shape=(12,1,n),val=0)
        #varTip = self.create_output('vt',shape=(12,1,n),val=0)
        # also can't create s_vec in the loop
        s_vec = self.create_output('s_vec',shape=(3,1,n),val=np.zeros((3,1,n)))

        Res = self.create_output('Res',shape=(num_variables,n),val=0)

        one = self.declare_variable('one',val=1)
        
        # section residual
        for i in range(0, n):
            if i <= n - 2:
                # rows 0-2: strain-displacement (ASW, Eq. 48, page 12)
                s_vec[1,0,i] = csdl.expand(one,(1,1,1),'i->ijk')
                tempVector = csdl.reshape(csdl.reshape(s_vec[:,:,i], new_shape=(3,1)) + 0.5*(strainsCSN[:, i] + strainsCSN[:, i + 1]), new_shape=(3))
                
                # rows 0-3: ------------------Compatibility Equations------------------
                #Res[0:3, i + 1] = R_prec[0:3] * (
                #        r[:, i + 1] - r[:, i] - delta_s0[i] * csdl.matmat(Ta[i][:, :].T, tempVector) + csdl.matmat(
                #    damp, ((u[:, i + 1] - u[:, i]) - csdl.cross((0.5 * (omega[:, i + 1] + omega[:, i])),
                #                                           (r[:, i + 1] - r[:, i])))))
                collapsed_Ta = csdl.reshape(Ta[:,:,i], new_shape=(3,3))
                t1 =  csdl.expand(delta_s0[i],(3)) * csdl.matvec(csdl.transpose(collapsed_Ta), tempVector)
                collapsed_omega_i_1 = csdl.reshape(omega[:,i+1], new_shape=(3))
                collapsed_omega_i = csdl.reshape(omega[:,i], new_shape=(3))
                collapsed_r_term = csdl.reshape(r[:,i+1] - r[:,i], new_shape=(3))
                collapsed_u_term = csdl.reshape(u[:,i+1] - u[:,i], new_shape=(3))
                inner_t2 = collapsed_u_term - csdl.cross((0.5*(collapsed_omega_i_1 + collapsed_omega_i)), collapsed_r_term, axis=0)
                t2 = csdl.matvec(damp, inner_t2)

                Res[0:3,i+1] = csdl.expand(R_prec[0:3]*(collapsed_r_term - t1 + t2), (3,1),'i->ijk')
                
                
                # rows 3-5: moment-curvature relationship (ASW, Eq. 54, page 13)
                #Res[3:6, i + 1] = R_prec[3:6] * (csdl.matmat(Ka[i][:, :], (theta[:, i + 1] - theta[:, i])) - csdl.matmat(K0a[i, :, :], (
                #        theta0[:, i + 1] - theta0[:, i])) - 0.25 * csdl.matmat((Einv[i][:, :] + Einv[i + 1][:, :]), (
                #        Mcsnp[:, i] + Mcsnp[:, i + 1])) * delta_s[i] + csdl.matmat(damp, (
                #        csdl.matmat(Ka[i][:, :], (damp_MK[:, i + 1] - damp_MK[:, i])) + 0.5 * csdl.matmat((
                #        K[i + 1][:, :] - K[i][:, :]), (damp_MK[:, i + 1] + damp_MK[:, i])))))
                collapsed_Ka = csdl.reshape(Ka[:,:,i], new_shape=(3,3))
                collapsed_theta_1 = csdl.reshape(theta[:,i+1], new_shape=(3))
                collapsed_theta = csdl.reshape(theta[:,i], new_shape=(3))
                t_1 = csdl.matvec(collapsed_Ka, (collapsed_theta_1 - collapsed_theta))

                collapsed_K0a = csdl.reshape(K0a[i,:,:], new_shape=(3,3))
                t_2 = csdl.matvec(collapsed_K0a, (collapsed_theta_1 - collapsed_theta))

                collapsed_Einv = csdl.reshape(Einv[:,:,i], new_shape=(3,3))
                collapsed_Einv_1 = csdl.reshape(Einv[:,:,i+1], new_shape=(3,3))
                collapsed_Mcsnp = csdl.reshape(Mcsnp[:,i], new_shape=(3))
                collapsed_Mcsnp_1 = csdl.reshape(Mcsnp[:,i+1], new_shape=(3))
                t_3 = 0.25 * csdl.matvec((collapsed_Einv + collapsed_Einv_1), (collapsed_Mcsnp + collapsed_Mcsnp_1)) * csdl.expand(delta_s[i],(3))

                collapsed_damp_MK_1 = csdl.reshape(damp_MK[:,i+1], new_shape=(3))
                collapsed_damp_MK = csdl.reshape(damp_MK[:,i], new_shape=(3))
                inner_t4_term_1 = csdl.matvec(collapsed_Ka, (collapsed_damp_MK_1 - collapsed_damp_MK))
                collapsed_K_1 = csdl.reshape(K[:,:,i+1], new_shape=(3,3))
                collapsed_K = csdl.reshape(K[:,:,i], new_shape=(3,3))
                inner_t4_term_2 = 0.5*csdl.matvec((collapsed_K_1 - collapsed_K), (collapsed_damp_MK_1 + collapsed_damp_MK))
                t_4 = csdl.matvec(damp, (inner_t4_term_1 + inner_t4_term_2))

                Res[3:6,i+1] = csdl.expand(R_prec[3:6]*(t_1 - t_2 - t_3 + t_4), (3,1),'i->ijk')

                
                # rows 6-8: force equilibrium (ASW, Eq. 56, page 13)
                collapsed_F = csdl.reshape(F[:,i], new_shape=(3))
                collapsed_F_1 = csdl.reshape(F[:,i+1], new_shape=(3))
                collapsed_f = csdl.reshape(f[:,i], new_shape=(3))
                ex_delta_s = csdl.expand(delta_s[i], (3))
                collapsed_delta_Fapplied = csdl.reshape(delta_Fapplied[:, i], new_shape=(3))
                Res[6:9, i] = csdl.expand(R_prec[6:9]*(collapsed_F_1 - collapsed_F + (collapsed_f*ex_delta_s) + collapsed_delta_Fapplied), (3,1),'i->ijk')
                
                """
                # rows 9-11: moment equilibrium (ASW, Eq. 55, page 13)
                Res[9:12, i] = R_prec[9:12] * (
                        M[:, i + 1] - M[:, i] + m[:, i] * delta_s[i] + delta_Mapplied[:, i] + csdl.cross(delta_r[:, i],
                                                                                                    Fav[:, i]))
                
                # Rows 12-14
                Res[12:15, i] = (u[:, i] - rDot[:, i])

                # Rows 15-17 (UNS, Eq. 2, page 4);
                Res[15:18, i] = (omega[:, i] - csdl.matmat(csdl.matmat(T[i][:, :].T, K[i][:, :]), thetaDot[:, i]))

            else:

                Res[12:15, i] = (u[:, i] - rDot[:, i])
                Res[15:18, i] = (
                        omega[:, i] - csdl.matmat(T[i][:, :].T, csdl.matmat(K[i][:, :], thetaDot[:, i])))
                
                # BOUNDARY CONDITIONS *****************************************
                BCroot = bc[element]['root']
                BCtip = bc[element]['tip']
                # potential variables to be set as bc
                # varRoot = SX.sym('vr', 12, 1)
                # varTip = SX.sym('vt', 12, 1)

                varRoot[i,0:3] = r[:, 0]
                varRoot[i,3:6] = theta[:, 0]
                varRoot[i,6:9] = F[:, 0]
                varRoot[i,9:12] = M[:, 0]

                varTip[i,0:3] = r[:, i]
                varTip[i,3:6] = theta[:, i]
                varTip[i,6:9] = F[:, i]
                varTip[i,9:12] = M[:, i]

                # indices that show which variables are to be set as bc (each will return 6 indices)
                indicesRoot_ = (~(BCroot == 8888))
                indicesTip_ = (~(BCtip == 8888))

                indicesRoot = []
                indicesTip = []
                for k in range(0, len(indicesRoot_)):
                    if indicesRoot_[k]:
                        indicesRoot.append(k)
                    if indicesTip_[k]:
                        indicesTip.append(k)
                # root
                Res[0:3, 0] = R_prec[12:15]*(varRoot[i,indicesRoot[0:3]] - BCroot[indicesRoot[0:3]])
                Res[3:6, 0] = R_prec[15:18]*(varRoot[i,indicesRoot[3:6]] - BCroot[indicesRoot[3:6]])
                # tip
                Res[6:9, i] = R_prec[18:21]*(varTip[i,indicesTip[0:3]] - BCtip[indicesTip[0:3]])
                Res[9:12, i] = R_prec[21:24]*(varTip[i,indicesTip[3:6]] - BCtip[indicesTip[3:6]])

        # endsection
        # return reshape(Res, (18 * n, 1))
        """