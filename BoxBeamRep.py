import numpy as np

n = 16

# params
E = 69E9
G = 1E20
rho = 2700
seq = np.array([3, 1, 2]) # fuselage beam

# cs params (all vectors of length n)
h = 0.5*np.ones((n))
w = 3*np.ones((n))
t_left = 0.1*np.ones((n))
t_top = 0.1*np.ones((n))
t_right = 0.1*np.ones((n))
t_bot = 0.1*np.ones((n))



# region Box-beam cross-section 4 parts
# Segments 1:
A_sect1 = t_top * w
rho_sect1 = rho
E_sect1 = E
cg_sect1_y = 0
cg_sect1_z = (h - t_top) / 2
Ixx_sect1 = (w * t_top ** 3) / 12
Izz_sect1 = (t_top * w ** 3) / 12
# Segments 2:
A_sect2 = t_left * (h - t_top - t_bot)
rho_sect2 = rho
E_sect2 = E
cg_sect2_y = (t_left - w) / 2
cg_sect2_z = h / 2 - t_top - (h - t_top - t_bot) / 2
Ixx_sect2 = (t_left * (h - t_top - t_bot) ** 3) / 12
Izz_sect2 = ((h - t_top - t_bot) * t_left ** 3) / 12
# Segments 3:
A_sect3 = t_bot * w
rho_sect3 = rho
E_sect3 = E
cg_sect3_y = 0
cg_sect3_z = (t_bot - h) / 2
Ixx_sect3 = (w * t_bot ** 3) / 12
Izz_sect3 = (t_bot * w ** 3) / 12
# Segments 4:
A_sect4 = t_right * (h - t_top - t_bot)
E_sect4 = E
rho_sect4 = rho
cg_sect4_y = (w - t_right) / 2
cg_sect4_z = h / 2 - t_top - (h - t_top - t_bot) / 2
Ixx_sect4 = (t_right * (h - t_top - t_bot) ** 3) / 12
Izz_sect4 = ((h - t_top - t_bot) * t_right ** 3) / 12
# endregion


# region offsets from the beam axis
e_cg_x = (cg_sect1_y * A_sect1 * E_sect1 +
            cg_sect2_y * A_sect2 * E_sect2 +
            cg_sect3_y * A_sect3 * E_sect3 +
            cg_sect4_y * A_sect4 * E_sect4) / \
            (A_sect1 * E_sect1 +
            A_sect2 * E_sect2 +
            A_sect3 * E_sect3 +
            A_sect4 * E_sect4)
e_cg_z = (cg_sect1_z * A_sect1 * E_sect1 +
            cg_sect2_z * A_sect2 * E_sect2 +
            cg_sect3_z * A_sect3 * E_sect3 +
            cg_sect4_z * A_sect4 * E_sect4) / \
            (A_sect1 * E_sect1 +
            A_sect2 * E_sect2 +
            A_sect3 * E_sect3 +
            A_sect4 * E_sect4)
n_ea = e_cg_z
c_ea = e_cg_x
n_ta = np.zeros((n))
c_ta = np.zeros((n))
# endregion



# region bending and torsional stiffness
# parallel axis theorem
Ixx = Ixx_sect1 + A_sect1 * (cg_sect1_z - e_cg_z) ** 2 + \
        Ixx_sect2 + A_sect2 * (cg_sect2_z - e_cg_z) ** 2 + \
        Ixx_sect3 + A_sect3 * (cg_sect3_z - e_cg_z) ** 2 + \
        Ixx_sect4 + A_sect4 * (cg_sect4_z - e_cg_z) ** 2
Izz = Izz_sect1 + A_sect1 * (cg_sect1_y - e_cg_x) ** 2 + \
        Izz_sect2 + A_sect2 * (cg_sect2_y - e_cg_x) ** 2 + \
        Izz_sect3 + A_sect3 * (cg_sect3_y - e_cg_x) ** 2 + \
        Izz_sect4 + A_sect4 * (cg_sect4_y - e_cg_x) ** 2
Ixz = -(A_sect1 * (cg_sect1_y - e_cg_x) * (cg_sect1_z - e_cg_z) +
        A_sect2 * (cg_sect2_y - e_cg_x) * (cg_sect2_z - e_cg_z) +
        A_sect3 * (cg_sect3_y - e_cg_x) * (cg_sect3_z - e_cg_z) +
        A_sect4 * (cg_sect4_y - e_cg_x) * (cg_sect4_z - e_cg_z))
J = 2 * (((h - t_top / 2 - t_bot / 2) * (w - t_right / 2 - t_left / 2)) ** 2) / \
        (((w - t_right / 2 - t_left / 2) / (0.5 * t_top + 0.5 * t_bot)) +
        ((h - t_top / 2 - t_bot / 2) / (0.5 * t_right + 0.5 * t_left)))

EIxx = E * Ixx
EIzz = E * Izz
EIxz = E * Ixz
GJ = G * J
# endregion


# region axial and shear stiffness
GKn = G / 1.2 * np.ones(n)
GKc = G / 1.2 * np.ones(n)
EA = E_sect1 * A_sect1 + E_sect2 * A_sect2 + E_sect3 * A_sect3 + E_sect4 * A_sect4
# endregion


# region mass properties
A = A_sect1 + A_sect2 + A_sect3 + A_sect4
mu = np.zeros((n - 1))
for i in range(n - 1):
    A1 = A[i]
    A2 = A[i + 1]
    mu[i] = rho * 1 / 3 * (A1 + A2 + np.sqrt(A1 * A2))
# endregion



# region delta_r_CG
m_cg_x = (cg_sect1_y * A_sect1 * rho_sect1 +
            cg_sect2_y * A_sect2 * rho_sect2 +
            cg_sect3_y * A_sect3 * rho_sect3 +
            cg_sect4_y * A_sect4 * rho_sect4) / \
            (A_sect1 * rho_sect1 +
            A_sect2 * rho_sect2 +
            A_sect3 * rho_sect3 +
            A_sect4 * rho_sect4)
m_cg_z = (cg_sect1_z * A_sect1 * rho_sect1 +
            cg_sect2_z * A_sect2 * rho_sect2 +
            cg_sect3_z * A_sect3 * rho_sect3 +
            cg_sect4_z * A_sect4 * rho_sect4) / \
            (A_sect1 * rho_sect1 +
            A_sect2 * rho_sect2 +
            A_sect3 * rho_sect3 +
            A_sect4 * rho_sect4)
# column i is the position of the CG on the cross-section i, relative to the csn origin, expressed in xyz
delta_r_CG = np.zeros((n - 1, 3))
for i in range(n - 1):
    delta_r_CG[i, 1] = (m_cg_x[i] + m_cg_x[i + 1]) / 2
    delta_r_CG[i, 2] = (m_cg_z[i] + m_cg_z[i + 1]) / 2
# endregion



# region shear terms
A_inner = ((h - t_top / 2 - t_bot / 2) * (w - t_right / 2 - t_left / 2))
Q_max_z = w * t_top * (h / 2 - e_cg_z - t_top / 2) + \
            0.5 * t_right * (h / 2 - t_top - e_cg_z) ** 2 + \
            0.5 * t_left * (h / 2 - t_top - e_cg_z) ** 2
Q_max_x = h * t_left * (w / 2 - e_cg_x - t_left / 2) + \
            0.5 * t_top * (w / 2 - t_left - e_cg_x) ** 2 + \
            0.5 * t_bot * (w / 2 - t_left - e_cg_x) ** 2
# endregion




D = np.zeros((3,3,n))
for i in range(n):
    D[0,0,i] = 0
    D[0,1,i] = -n_ea[i]
    D[0,2,i] = 0
    D[1,0,i] = n_ta[0]
    D[1,1,i] = 0
    D[1,2,i] = -c_ta[0]
    D[2,0,i] = 0
    D[2,1,i] = c_ea[0]
    D[2,2,i] = 0



oneover = np.zeros((3,3,n))
for i in range(n):
    oneover[0,0,i] = 1 / GKc[i]
    oneover[0,1,i] = 0
    oneover[0,2,i] = 0
    oneover[1,0,i] = 0
    oneover[1,1,i] = 1 / EA[i]
    oneover[1,2,i] = 0
    oneover[2,0,i] = 0
    oneover[2,1,i] = 0
    oneover[2,2,i] = 1 / GKn[i]


i_matrix = np.zeros((3,3,n-1))
for i in range(n - 1):
    i_matrix[0,0,i] = mu[i]
    i_matrix[0,1,i] = 0
    i_matrix[0,2,i] = 0
    i_matrix[1,0,i] = 0
    i_matrix[1,1,i] = mu[i]
    i_matrix[1,2,i] = 0
    i_matrix[2,0,i] = 0
    i_matrix[2,1,i] = 0
    i_matrix[2,2,i] = mu[i]


delta_r_CG_tilde = np.zeros((3,3,n - 1))
for i in range(n - 1):
    drCG = delta_r_CG[i, :]
    delta_r_CG_tilde[0,0,i] = 0
    delta_r_CG_tilde[0,1,i] = -drCG[2]
    delta_r_CG_tilde[0,2,i] = drCG[1]
    delta_r_CG_tilde[1,0,i] = drCG[2]
    delta_r_CG_tilde[1,1,i] = 0
    delta_r_CG_tilde[1,2,i] = -drCG[0]
    delta_r_CG_tilde[2,0,i] = -drCG[1]
    delta_r_CG_tilde[2,1,i] = drCG[0]
    delta_r_CG_tilde[2,2,i] = 0



Emat = np.zeros((3, 3, n))
Einv = np.zeros((3, 3, n))
# E matrix where each 3x3 corresponds to cross section i at node i
# E has the following form:
# E = [EIcc  EIcs     EIcn;
#      0     GJ       EIsn;
#      0     GJ       EIsn;
#      0     0        EInn]
for i in range(n):
    Emat[0,0,i] = EIxx[i]
    Emat[0,1,i] = 0
    Emat[0,2,i] = EIxz[i]
    Emat[1,0,i] = 0
    Emat[1,1,i] = GJ[i]
    Emat[1,2,i] = 0
    Emat[2,0,i] = EIxz[i]
    Emat[2,1,i] = 0
    Emat[2,2,i] = EIzz[i]

    Einv[:,:,i] = np.linalg.inv(Emat[:,:,i])





# K0a
th0 = np.ones((3,n))*1 # initial lengths?
K0a = np.zeros((n - 1, 3, 3))
K = np.zeros((n, 3, 3))
for i in range(n):
    if (seq == np.array([3,1,2])).all():  # (fuselage) seq=312
        K[i, :, :] = np.asarray([[np.cos(th0[1, i]),0,-np.cos(th0[0, i])*np.sin(th0[1, i])],
            [0,1,np.sin(th0[0, i])],
            [np.sin(th0[1, i]),
             0, np.cos(th0[0, i])*np.cos(th0[1, i])]])
        if i >= 1:
            K0a[i - 1, :, :] = (K[i, :, :] + K[i - 1, :, :]) / 2
    elif (seq == np.array([132])).all():  # (wing) seq=132
        K[i, :, :] = np.array([[np.cos(th0[2, i])*np.cos(th0[1, i]),0,-np.sin(th0[1, i])],
                    [-np.sin(th0[2, i]),1,0],
                    [np.cos(th0[2, i])*np.sin(th0[1, i]),0,np.cos(th0[1, i])]])
        if i >= 1:
            K0a[i - 1, :, :] = (K[i, :, :] + K[i - 1, :, :]) / 2
    else:
        raise IOError