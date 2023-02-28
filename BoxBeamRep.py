import numpy as np

# params
E = 69E9
G = 1E20
rho = 2700



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

self.symbolic_expressions['A_segments'] = {'Top': A_sect1,
                                                   'Left': A_sect2,
                                                   'Bot': A_sect3,
                                                   'Right': A_sect4}
# endregion