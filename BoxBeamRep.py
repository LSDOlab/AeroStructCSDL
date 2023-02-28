import numpy as np

# params
E = 69E9
G = 1E20
rho = 2700
seq = np.array([3, 1, 2]) # fuselage beam
n = 16




"""
if np.array_equal(seq, np.array([3, 1, 2])):  # fuselage beam
    span_var = 0
else:
    span_var = 1

# Design Variables
cs = SX.sym(self.options['name'] + 'cs', 6 * n_dvcs)
self.symbolic_expressions['cs'] = cs

# isolate important variables from cs
h_seed = cs[0:n_dvcs]
w_seed = cs[n_dvcs:2 * n_dvcs]
t_left_seed = cs[2 * n_dvcs:3 * n_dvcs]
t_top_seed = cs[3 * n_dvcs:4 * n_dvcs]
t_right_seed = cs[4 * n_dvcs:5 * n_dvcs]
t_bot_seed = cs[5 * n_dvcs:6 * n_dvcs]

# construct the augmented set
h = SX.zeros(n, 1)
w = SX.zeros(n, 1)
t_left = SX.zeros(n, 1)
t_top = SX.zeros(n, 1)
t_right = SX.zeros(n, 1)
t_bot = SX.zeros(n, 1)

        # Lay up the augmented section set with the available cross-section information:
        J = 0
        for i in range(0, n):
            if self.options['section_characteristics'][i] == SectionType.CS:
                h[i] = h_seed[J]
                w[i] = w_seed[J]
                t_left[i] = t_left_seed[J]
                t_top[i] = t_top_seed[J]
                t_right[i] = t_right_seed[J]
                t_bot[i] = t_bot_seed[J]
                J += 1
            else:
                if i == n - 1 or J == n_dvcs:  # Last element
                    h[i] = h_seed[-1]
                    w[i] = w_seed[-1]
                    t_left[i] = t_left_seed[-1]
                    t_top[i] = t_top_seed[-1]
                    t_right[i] = t_right_seed[-1]
                    t_bot[i] = t_bot_seed[-1]
                else:
                    h_prev = h_seed[J-1]
                    w_prev = w_seed[J-1]
                    t_left_prev = t_left_seed[J-1]
                    t_top_prev = t_top_seed[J-1]
                    t_right_prev = t_right_seed[J-1]
                    t_bot_prev = t_bot_seed[J-1]

                    h_next = h_seed[J]
                    w_next = w_seed[J]
                    t_left_next = t_left_seed[J]
                    t_top_next = t_top_seed[J]
                    t_right_next = t_right_seed[J]
                    t_bot_next = t_bot_seed[J]

                    y_prev = cs_r0[span_var, J - 1]
                    y_current = r0[span_var, i]
                    y_next = cs_r0[span_var, J]

                    h[i] = h_next * (y_current - y_prev) / (y_next - y_prev) + h_prev * (
                                1 - (y_current - y_prev) / (y_next - y_prev))
                    w[i] = w_next * (y_current - y_prev) / (y_next - y_prev) + w_prev * (
                                1 - (y_current - y_prev) / (y_next - y_prev))

                    t_left[i] = t_left_next * (y_current - y_prev) / (y_next - y_prev) + t_left_prev * (
                            1 - (y_current - y_prev) / (y_next - y_prev))
                    t_top[i] = t_top_next * (y_current - y_prev) / (y_next - y_prev) + t_top_prev * (
                            1 - (y_current - y_prev) / (y_next - y_prev))

                    t_right[i] = t_right_next * (y_current - y_prev) / (y_next - y_prev) + t_right_prev * (
                            1 - (y_current - y_prev) / (y_next - y_prev))
                    t_bot[i] = t_bot_next * (y_current - y_prev) / (y_next - y_prev) + t_bot_prev * (
                            1 - (y_current - y_prev) / (y_next - y_prev))
                    pass

"""






# cs params
h = 
w = 
t_left = 
t_top = 
t_right = 
t_bot = 





















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