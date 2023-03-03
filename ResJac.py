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
        self.parameters.declare('bc')
        self.parameters.declare('seq')
    def define(self):
        n = self.parameters['num_nodes']
        num_variables = self.parameters['num_variables']
        bc = self.parameters['bc'] # boundary conditions
        g = self.declare_variable('g',shape=(3),val=np.array([0,0,0]))
        seq = self.parameters['seq']

        x = self.declare_variable('x',shape=(num_variables,n),val=0) # state vector
        xd = self.declare_variable('xd',shape=(num_variables,n),val=0) # time derivatives of state vector
        xac = self.declare_variable('xac',shape=(num_variables),val=0) # aircraft state vector
        R_prec = self.declare_variable('R_prec',shape=(24),val=1) # numerical scaling
        
        # read x
        r = x[0:3, :] # nodal positions (x,y,z)
        self.register_output('r',r)
        theta = x[3:6, :] # nodal orientations
        self.register_output('theta',theta)
        F = x[6:9, :] # nodal beam-stress forces
        self.register_output('F',F)
        M = x[9:12, :] # nodal beam stress moments
        self.register_output('M',M)
        u = x[12:15, :] # nodal velocities (x,y,z)
        self.register_output('u',u)
        omega = x[15:18, :] # nodal rotations
        self.register_output('omega',omega)

        
        # read xDot (time derivatives)
        rDot = xd[0:3, :]
        self.register_output('rDot',rDot)
        thetaDot = xd[3:6, :]
        #self.register_output('thetaDot',thetaDot)
        #FDot = xd[6:9, :]
        #self.register_output('FDot',FDot)
        #MDot = xd[9:12, :]
        #self.register_output('MDot',MDot)
        uDot = xd[12:15, :]
        self.register_output('uDot',uDot) 
        omegaDot = xd[15:18, :]
        self.register_output('omegaDot',omegaDot)
        
        # read the aircraft states
        # R = xac[0:3] # aircraft position (X,Y,Z)
        # U = xac[3:6] # aircraft velocity
        # self.register_output('U',U)
        A0 = xac[6:9]
        self.register_output('A0',A0)
        THETA = xac[9:12]
        self.register_output('THETA',THETA)
        OMEGA = xac[12:15]
        self.register_output('OMEGA',OMEGA)
        ALPHA0 = xac[15:18]
        self.register_output('ALPHA0',ALPHA0)
        
        # forces and moments
        f_aero = self.declare_variable('f_aero',shape=(3,n-1),val=0) # distributed aero forces
        m_aero = self.declare_variable('m_aero',shape=(3,n-1),val=0) # distributed aero moments
        delta_Fapplied = self.declare_variable('delta_Fapplied',shape=(3,n-1),val=0) # point loads
        delta_Mapplied = self.declare_variable('delta_Mapplied',shape=(3,n-1),val=0) # point moments
        
        # read the stick model properties
        mu = self.declare_variable('mu',shape=(n-1)) # vector of mass/length
        theta0 = self.declare_variable('theta0',shape=(n),val=0) # unloaded nodal orientations
        K0a = self.declare_variable('K0a',shape=(n-1,3,3))
        delta_s0 = self.declare_variable('delta_s0',shape=(n-1))
        i_matrix = self.declare_variable('i_matrix',shape=(3,3,n-1))
        delta_rCG_tilde = self.declare_variable('delta_rCG_tilde',shape=(3,3,n-1))
        Einv = self.declare_variable('Einv',shape=(3,3,n))
        D = self.declare_variable('D',shape=(3,3,n))
        oneover = self.declare_variable('oneover',shape=(3,3,n))

        self.add(calc_a_cg(num_nodes=n),name='calc_a_cg')
        a_cg = self.declare_variable('aCG',shape=(3,n-1))
        self.print_var(a_cg)
        
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




        # convert options
        t_epsilon = self.declare_variable('t_epsilon',shape=(1,1))
        t_gamma = self.declare_variable('t_gamma',shape=(1,1))
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
        varRoot = self.create_output('vr',shape=(12,n),val=0)
        varTip = self.create_output('vt',shape=(12,n),val=0)
        # also can't create s_vec in the loop
        s_vec = self.create_output('s_vec',shape=(3,n),val=np.zeros((3,n)))

        Res = self.create_output('Res',shape=(num_variables,n),val=0)

        one = self.declare_variable('one',val=1)
        zero = self.declare_variable('zero',val=0)
        
        # section residual
        for i in range(0, n):
            if i <= n - 2:
                # rows 0-2: strain-displacement (ASW, Eq. 48, page 12)
                s_vec[1,i] = csdl.expand(one,(1,1),'i->ij')
                tempVector = csdl.reshape(s_vec[:,i] + 0.5*(strainsCSN[:, i] + strainsCSN[:, i + 1]), new_shape=(3)) # (0,1,0)
                
                # rows 0-3: ------------------Compatibility Equations------------------
                collapsed_Ta = csdl.reshape(Ta[:,:,i], new_shape=(3,3)) # eye ?
                t1 = csdl.expand(delta_s0[i],(3)) * csdl.matvec(csdl.transpose(collapsed_Ta), tempVector)
                collapsed_omega_term = csdl.reshape(omega[:,i+1] + omega[:,i], new_shape=(3))
                collapsed_r_term = csdl.reshape(r[:,i+1] - r[:,i], new_shape=(3))
                collapsed_u_term = csdl.reshape(u[:,i+1] - u[:,i], new_shape=(3))
                inner_t2 = collapsed_u_term - csdl.cross(0.5*collapsed_omega_term, collapsed_r_term, axis=0)
                t2 = csdl.matvec(damp, inner_t2)

                Res[0:3,i+1] = csdl.expand(R_prec[0:3]*(collapsed_r_term - t1 + t2), (3,1),'i->ij')
                
                
                # rows 3-5: moment-curvature relationship (ASW, Eq. 54, page 13)
                #Res[3:6, i + 1] = R_prec[3:6] * (csdl.matmat(Ka[i][:, :], (theta[:, i + 1] - theta[:, i])) - csdl.matmat(K0a[i, :, :], (
                #        theta0[:, i + 1] - theta0[:, i])) - 0.25 * csdl.matmat((Einv[i][:, :] + Einv[i + 1][:, :]), (
                #        Mcsnp[:, i] + Mcsnp[:, i + 1])) * delta_s[i] + csdl.matmat(damp, (
                #        csdl.matmat(Ka[i][:, :], (damp_MK[:, i + 1] - damp_MK[:, i])) + 0.5 * csdl.matmat((
                #        K[i + 1][:, :] - K[i][:, :]), (damp_MK[:, i + 1] + damp_MK[:, i])))))
                collapsed_Ka = csdl.reshape(Ka[:,:,i], new_shape=(3,3))
                collapsed_theta_term = csdl.reshape(theta[:,i+1] - theta[:,i], new_shape=(3))
                t_1 = csdl.matvec(collapsed_Ka, collapsed_theta_term)
                collapsed_K0a = csdl.reshape(K0a[i,:,:], new_shape=(3,3))
                t_2 = csdl.matvec(collapsed_K0a, collapsed_theta_term)
                collapsed_Einv_term = csdl.reshape(Einv[:,:,i] + Einv[:,:,i+1], new_shape=(3,3))
                collapsed_Mcsnp_term = csdl.reshape(Mcsnp[:,i] + Mcsnp[:,i+1], new_shape=(3))
                t_3 = 0.25 * csdl.matvec(collapsed_Einv_term, collapsed_Mcsnp_term) * csdl.expand(delta_s[i],(3))
                collapsed_damp_MK_1 = csdl.reshape(damp_MK[:,i+1], new_shape=(3))
                collapsed_damp_MK = csdl.reshape(damp_MK[:,i], new_shape=(3))
                inner_t4_term_1 = csdl.matvec(collapsed_Ka, (collapsed_damp_MK_1 - collapsed_damp_MK))
                collapsed_K_term = csdl.reshape(K[:,:,i+1] - K[:,:,i], new_shape=(3,3))
                inner_t4_term_2 = 0.5*csdl.matvec(collapsed_K_term, (collapsed_damp_MK_1 + collapsed_damp_MK))
                t_4 = csdl.matvec(damp, (inner_t4_term_1 + inner_t4_term_2))

                Res[3:6,i+1] = csdl.expand(R_prec[3:6]*(t_1 - t_2 - t_3 + t_4), (3,1),'i->ij')

                
                # rows 6-8: force equilibrium (ASW, Eq. 56, page 13)
                collapsed_F_term = csdl.reshape(F[:,i+1] - F[:,i], new_shape=(3)) # correct 0
                collapsed_f = csdl.reshape(f[:,i], new_shape=(3)) # correct 0
                ex_delta_s = csdl.expand(delta_s[i], (3)) # correct maybe ~ 1E-19
                collapsed_delta_Fapplied = csdl.reshape(delta_Fapplied[:, i], new_shape=(3)) # correct 0
                Res[6:9,i] = csdl.expand(R_prec[6:9]*(collapsed_F_term + (collapsed_f*ex_delta_s) + collapsed_delta_Fapplied), (3,1),'i->ij')

                self.print_var((collapsed_f))
                
                
                # rows 9-11: moment equilibrium (ASW, Eq. 55, page 13)
                collapsed_M = csdl.reshape(M[:,i], new_shape=(3))
                collapsed_M_1 = csdl.reshape(M[:,i+1], new_shape=(3))
                collapsed_m = csdl.reshape(m[:,i], new_shape=(3))
                collapsed_delta_Mapplied = csdl.reshape(delta_Mapplied[:, i], new_shape=(3))
                collapsed_delta_r = csdl.reshape(delta_r[:,i], new_shape=(3))
                collapsed_Fav = csdl.reshape(Fav[:, i], new_shape=(3))
                Res[9:12,i] = csdl.expand(R_prec[9:12]*(collapsed_M_1 - collapsed_M + collapsed_m*ex_delta_s + collapsed_delta_Mapplied + csdl.cross(collapsed_delta_r, collapsed_Fav, axis=0)), (3,1),'i->ij')


                # Rows 12-14
                Res[12:15,i] = (u[:,i] - rDot[:, i])


                # Rows 15-17 (UNS, Eq. 2, page 4);
                collapsed_K = csdl.reshape(K[:,:,i], new_shape=(3,3))
                collapsed_T = csdl.reshape(T[:,:,i], new_shape=(3,3))
                Res[15:18, i] = omega[:, i] - csdl.expand(csdl.matvec(csdl.matmat(csdl.transpose(collapsed_T), collapsed_K), csdl.reshape(thetaDot[:,i], new_shape=(3))), (3,1),'i->ij')


            else:

                Res[12:15, i] = (u[:, i] - rDot[:, i])
                
                collapsed_T_transpose = csdl.transpose(csdl.reshape(T[:,:,i], new_shape=(3,3)))
                collapsed_K = csdl.reshape(K[:,:,i], new_shape=(3,3))
                collapsed_thetaDot = csdl.reshape(thetaDot[:,i], new_shape=(3))

                Res[15:18, i] = omega[:, i] - csdl.expand(csdl.matvec(collapsed_T_transpose, csdl.matvec(collapsed_K,collapsed_thetaDot)), (3,1),'i->ij')
                

                # BOUNDARY CONDITIONS *****************************************
                BCroot = bc['root']
                BCtip = bc['tip']
                # potential variables to be set as bc

                varRoot[0:3,i] = r[:, 0]
                varRoot[3:6,i] = theta[:, 0]
                varRoot[6:9,i] = F[:, 0]
                varRoot[9:12,i] = M[:, 0]

                varTip[0:3,i] = r[:, i]
                varTip[3:6,i] = theta[:, i]
                varTip[6:9,i] = F[:, i]
                varTip[9:12,i] = M[:, i]

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
                Res[0:3, 0] = csdl.expand(R_prec[12:15]*(csdl.reshape(varRoot[indicesRoot[0]:indicesRoot[3],i], new_shape=(3)) - BCroot[indicesRoot[0:3]]), (3,1),'i->ij')
                Res[3:6, 0] = csdl.expand(R_prec[15:18]*(csdl.reshape(varRoot[indicesRoot[2]:indicesRoot[5],i], new_shape=(3)) - BCroot[indicesRoot[3:6]]), (3,1),'i->ij') # 6 to 5
                # tip
                Res[6:9, i] = csdl.expand(R_prec[18:21]*(csdl.reshape(varTip[indicesRoot[0]:indicesRoot[3],i], new_shape=(3)) - BCtip[indicesTip[0:3]]), (3,1),'i->ij')
                Res[9:12, i] = csdl.expand(R_prec[21:24]*(csdl.reshape(varTip[indicesRoot[2]:indicesRoot[5],i], new_shape=(3)) - BCtip[indicesTip[3:6]]), (3,1),'i->ij') # 6 to 5