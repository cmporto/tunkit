import numpy as np







# Defining the converting factors
hartree_to_J       = 4.35974472220e-18
J_to_hartree       = 1 / hartree_to_J
hartree_to_cm      = 219474.63136320
cm_to_hartree      = 1 / hartree_to_cm
kcalmol_to_J       = 6.9477e-21
J_to_kcalmol       = 1 / kcalmol_to_J
hartree_to_kcalmol = hartree_to_J * J_to_kcalmol
cm_to_J            = 1.98644586e-23
J_to_cm            = 1 / cm_to_J
hbar               = 1.054571817E-34 # kg m^2 /s
hbar_au            = 1.0 # au
h_plank            = hbar * 2 * np.pi # kg m^2 /s
amu                = 1.66053906660E-27 # proton mass 1.67262192E-27 # kg
amu_to_au          = 1822.8885300626 
au_to_amu          = 1 / amu_to_au
speed_light        = 29979245800    # cm/s
bohr_to_ang        = 0.529177210544
ang_to_bohr        = 1 / bohr_to_ang
k_B_au             = 3.166811563e-6 # 1,380649e-23 J/K * J_to_hartree