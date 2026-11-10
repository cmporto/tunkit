from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import trapezoid as trapz
from scipy import linalg 
from tunkit.core import constants as const









class ResPlotMixin:

        #[TODO] The lateral of the plot is not showing the whole potential when the plot is below
    # the requested energy levels. This is a problem. I should probably just organize it for 
    # just looking at the potential and not the energy levels. As for example a percentage of 
    # the size of the barrier. And laterally to always show the whole potential there.
    def plot_symmetric_results(self, 
                            n_levels=4, 
                            red_mass=1.0, 
                            scale_factor=0.005, 
                            units='Hartree',
                            savefig_file='',
                            scaling_ylim_sup = 0.4,
                            scaling_ylim_inf = 1.03,
                            graph_title = 'Symmetric DWP',
                            leg_loc='',
                            plot_potential=False
                            ):
        """
        Plots the symmetric double well potential and wave functions using 
        a robust Finite Difference (grid) approach instead of Legendre reconstruction.
        """
        if not hasattr(self, 'h_vec') or len(self.h_vec) == 0:
            print("Error: No calculation results found.")
            return

        red_mass = red_mass * const.amu_to_au

        # last fit
        h = self.h_vec[-1]
        b = self.b_vec[-1]
        L = self.L_vec[-1]
        
        # grid
        N_grid = 2000
        epsilon = 1e-3 # Avoid singularity at boundaries
        y_grid = np.linspace(-np.pi/2 + epsilon, np.pi/2 - epsilon, N_grid)
        dy = y_grid[1] - y_grid[0]
        
        # [FIXED] function implemented, changed to method. Remove after testing
        # tan_y = np.tan(y_grid)
        # sin_y = np.sin(y_grid)
        # U_dim = h * (tan_y**2) + b * (sin_y**2)

        # V(y) = h * tan^2(y) + b * sin^2(y)
        U_dim = self._aps_sym_potential(y_grid, h, b)

        # clamp potential to avoid numerical infinities at edges
        U_dim = np.clip(U_dim, -1e9, 1e9)

        # build Hamiltonian matrix for finite difference
        # H = -d^2/dy^2 + V(y)
        # diagonal elements: 2/dy^2 + V_i
        # off-diagonal: -1/dy^2
        kin_coeff = 1.0 / (dy**2)
        
        diag = 2 * kin_coeff + U_dim
        off_diag = -kin_coeff * np.ones(N_grid - 1)
        

        # we only need the first n_levels
        evals_dim, evecs = linalg.eigh_tridiagonal(
            diag, off_diag, select='i', select_range=(0, n_levels-1)
        )
        
        # normalize wavefunction
        for i in range(len(evals_dim)):
            norm = np.sqrt(trapz(evecs[:, i]**2, y_grid))
            evecs[:, i] /= norm

        # conversion
        C_hartree = (const.hbar_au**2 * np.pi**2) / (8 * red_mass * L**2)
        
        x_ang = (2 * L * y_grid / np.pi) # Grid in Bohrs 
        
        potential_energies = (self.proj_energies-np.max(self.proj_energies)) # Shift so that the minimum is at 0

        if units == 'kcal/mol':
            conv = 1.0 * const.hartree_to_kcalmol
            y_label = 'Energy (kcal/mol)'
        elif units == 'cm-1':
            conv = 1.0 * const.hartree_to_cm
            y_label = 'Energy (cm$^{-1}$)'
        else:
            conv = 1.0 
            y_label = 'Energy (Hartree)'
            
        potential_energies_phys = potential_energies * conv
        U_phys = U_dim * C_hartree * conv
        E_phys = evals_dim * C_hartree * conv
        

        if not leg_loc:
            plt.figure(figsize=(9, 6))
        else:
            plt.figure(figsize=(8, 6))
        
        # maximum energy level to scale ylim
        e_max_plot = E_phys[-1]
        
        # minimum to scale
        u_min = np.min(U_phys)
        

        # [DEBUG] Plot the potential in order to compare it to the fitted potential. In our experience, even
        # if the fitted potential is not perfect, the parameters balance themselves out. For example, a wider
        # well but shallower or vice versa. 
        if plot_potential:
            # plt.scatter(self.reaction_path, potential_energies_phys, c='r', zorder=100, s=10) #  *const.bohr_to_ang
            plt.plot(self.reaction_path, potential_energies_phys, '-o', c='darkblue',  markerfacecolor='r', markeredgecolor='r', ms=3, label='Trig. potential') # *const.bohr_to_ang

        plt.plot(x_ang, U_phys, 'k-', linewidth=2, label='Potential')

        for i in range(len(E_phys)):
            E = E_phys[i]
            psi = evecs[:, i]
            
            # [FIXED] slightly decreasing the norm as n_levels increases
            amp_scale = scale_factor / (abs(E-u_min * 1.2)) 

            # normalize peak to 1 then scale
            psi_norm = psi / np.max(np.abs(psi))
            psi_plot = E + (psi_norm * amp_scale)
            
            plt.axhline(y=E, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            plt.plot(x_ang, psi_plot, linewidth=1.5, label=r'$\psi_{{{}}}$'.format(i)) #color=colors[i%len(colors)],
            plt.fill_between(x_ang, E, psi_plot, alpha=0.2) #color=colors[i%len(colors)],



        # settign ylim
        y_lim_inf = np.min(U_phys) * scaling_ylim_inf
        y_lim_sup = (np.max(E_phys) - np.min(U_phys)) * scaling_ylim_sup
        
        # check where the x corresopnds to the y to get xlim
        valid_indices = np.where(U_phys < np.max(y_lim_sup))[0]

        if len(valid_indices) > 0:
            plt.xlim(x_ang[valid_indices[0]], x_ang[valid_indices[-1]])
            # plt.ylim(y_lim_inf, y_lim_sup) 
            
            plt.ylim(u_min * scaling_ylim_inf, e_max_plot + (e_max_plot - u_min)* scaling_ylim_sup)

        plt.xlabel(r'Reaction coordinate (Bohr amu$^{1/2}$)')
        plt.ylabel(y_label)
        plt.title(graph_title)
        if not leg_loc:
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) 
        else:
            plt.legend(loc=leg_loc)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if savefig_file:
            plt.savefig(savefig_file, dpi=300)
        plt.show()








    # [TODO] plot results should be one, and maybe different support functions
    # [TODO] this method could be simplified a lot and separated into different methods that are also used elsewhere
    def plot_asymmetric_results(self, 
                            n_levels=4, 
                            red_mass=1.0, 
                            scale_factor=0.005, 
                            units='kcal/mol',
                            savefig_file='',
                            scaling_ylim_sup = 0.4,
                            scaling_ylim_inf = 1.05,
                            graph_title = 'Asymmetric DWP',
                            leg_loc='',
                            plot_potential=False):
        """
        Plots the double well potential, the last n energy levels, and their wave functions.
        
        Parameters:
        -----------
        n_levels : int
            Number of levels to plot (default 4).
        red_mass : float
            Reduced mass in AMU (must match what was used in calculation).
        scale_factor : float
            Scaling factor to make wave functions visible on the energy plot.
            Adjust this if wave functions are too big/small.
        units : str
            'Hartree', 'kcal/mol', or 'cm-1' for the Y-axis.
        """

        if not hasattr(self, 'h_vec') or len(self.h_vec) == 0:
            print("Error: No calculation results found. Run calc_asymmetric_energy_levels first.")
            return

        red_mass = red_mass * const.amu_to_au

        h = self.h_vec[-1]
        b = self.b_vec[-1]
        a = self.a_vec[-1]
        L = self.L_vec[-1]

        eig_vals, eig_vecs, y_grid = self.solve_asymmetric_finite_difference(h, b, a, n_levels=n_levels)

        c = np.sqrt(h)
        
        # U(y) 
        # tan_y = np.tan(y_grid)
        # sin_y = np.sin(y_grid)
        # cos_sq = np.cos(y_grid)**2
        # U_dim = (h * tan_y**2) + (c * sin_y / cos_sq) - (b * sin_y**2) + (a * sin_y)

        # [FIXED] changed to method should remove it after testing
        U_dim = self._aps_asym_potential(y_grid, h, b, a)

        # avoid infinity at edges
        U_dim = np.clip(U_dim, np.min(U_dim), np.max(eig_vals) + 10000)

        # x = (2 * L * y) / pi
        x_ang = (2 * L * y_grid / np.pi) #* 1e10 

        C_hartree = (const.hbar_au**2 * np.pi**2) / (8 * red_mass * L**2)
        
        # unit conversion
        if units == 'kcal/mol':
            conv = 1.0 * const.hartree_to_kcalmol
            y_label = 'Energy (kcal/mol)'
        elif units == 'cm-1':
            conv = 1.0 * const.hartree_to_cm
            y_label = 'Energy (cm$^{-1}$)'
        else: # Default Hartree
            conv = 1.0 
            y_label = 'Energy (Hartree)'
            
        potential_energies = (self.proj_energies-np.max(self.proj_energies))#*const.hartree_to_J # Shift so that the minimum is at 0
        potential_energies_phys = potential_energies * conv
        U_phys = U_dim * C_hartree * conv
        E_phys = eig_vals * C_hartree * conv

        # getting the well minimum to scale the wave functions
        u_min = np.min(U_phys)

        plt.figure(figsize=(8, 6))
        # potential
        plt.plot(x_ang, U_phys, 'k-', linewidth=2, label='Potential')

        for i in range(len(E_phys)):
            E = E_phys[i]
            psi = eig_vecs[:, i]
            
            # but for plotting, we just want it to look nice.
            psi_norm = psi / np.max(np.abs(psi))
            
            # [FIXED] slightly decreasing the norm as n_levels increases
            psi_plot = E + (psi_norm * scale_factor / (abs(E-u_min * 1.2)))

            # level, wavefunction and filling
            plt.axhline(y=E, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            plt.plot(x_ang, psi_plot, linewidth=1.5, label=r'$\psi_{{{}}}$'.format(i))
            plt.fill_between(x_ang, E, psi_plot, alpha=0.2)


        if plot_potential:
            plt.scatter(self.reaction_path, potential_energies_phys, c='r', zorder=100, s=10) # *const.bohr_to_ang
            plt.plot(self.reaction_path, potential_energies_phys, c='darkblue') # *const.bohr_to_ang


        y_lim_sup = (np.max(E_phys) - np.min(U_phys)) * scaling_ylim_sup
        y_lim_inf = np.min(U_phys) * scaling_ylim_inf
        valid_indices = np.where(U_phys < np.max(y_lim_sup))[0]
        if len(valid_indices) > 0:
            plt.xlim(x_ang[valid_indices[0]], x_ang[valid_indices[-1]])
            plt.ylim(y_lim_inf, y_lim_sup) #+ (E_phys[1]-E_phys[0])

        plt.xlabel(r'Reaction Coordinate (Bohr amu$^{1/2}$)')
        plt.ylabel(y_label)
        plt.title(graph_title)
        if leg_loc:
            plt.legend(loc=leg_loc)
        else:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if savefig_file:
            plt.savefig(savefig_file, dpi=300)
        plt.show()








    def plot_asymmetric_prob_dens(self, 
                            n_levels=4, 
                            red_mass=1.0, 
                            scale_factor=0.005, 
                            units='kcal/mol',
                            savefig_file='',
                            scaling_ylim_sup = 0.4,
                            scaling_ylim_inf = 1.05,
                            graph_title = 'Asymmetric DWP Probability Density',
                            leg_loc=''):
        """
        Plots the double well potential, the last n energy levels, and their 
        probability densities (|psi|^2).
        
        Parameters:
        -----------
        n_levels : int
            Number of levels to plot (default 4).
        red_mass : float
            Reduced mass in AMU (must match what was used in calculation).
        scale_factor : float
            Scaling factor to make probability densities visible on the energy plot.
        units : str
            'Hartree', 'kcal/mol', or 'cm-1' for the Y-axis.
        """

        if not hasattr(self, 'h_vec') or len(self.h_vec) == 0:
            print("Error: No calculation results found. Run calc_asymmetric_energy_levels first.")
            return
        
        red_mass = red_mass * const.amu_to_au

        h = self.h_vec[-1]
        b = self.b_vec[-1]
        a = self.a_vec[-1]
        L = self.L_vec[-1]

        eig_vals, eig_vecs, y_grid = self.solve_asymmetric_finite_difference(h, b, a, n_levels=n_levels)

        # [FIXED] reuse method symmetric potential. Remove after testing
        # tan_y = np.tan(y_grid)
        # sin_y = np.sin(y_grid)
        # cos_sq = np.cos(y_grid)**2
        # c = np.sqrt(h)
        # U_dim = (h * tan_y**2) + (c * sin_y / cos_sq) - (b * sin_y**2) + (a * sin_y)
        
        # U(y) 
        U_dim = self._aps_asym_potential(y_grid, h, b, a)

        U_dim = np.clip(U_dim, np.min(U_dim), np.max(eig_vals) + 10000)

        # in bohrs
        x_ang = (2 * L * y_grid / np.pi) #* 1e10 

        C_hartree = (const.hbar_au**2 * np.pi**2) / (8 * red_mass * L**2)
        
        # Unit conversion
        if units == 'kcal/mol':
            conv = 1.0 * const.hartree_to_kcalmol
            y_label = 'Energy (kcal/mol)'
        elif units == 'cm-1':
            conv = 1.0 * const.hartree_to_cm
            y_label = 'Energy (cm$^{-1}$)'
        else: # Default Hartree
            conv = 1.0 
            y_label = 'Energy (Hartree)'
            
        U_phys = U_dim * C_hartree * conv
        E_phys = eig_vals * C_hartree * conv

        plt.figure(figsize=(8, 6))
        
        # plot potential
        plt.plot(x_ang, U_phys, 'k-', linewidth=2, label='Potential')

        # Getting the well minimum to scale the probability densities
        u_min = np.min(U_phys)

        for i in range(len(E_phys)):
            E = E_phys[i]
            psi = eig_vecs[:, i]
            
            prob_density = psi**2
            
            prob_norm = prob_density / np.max(prob_density)
            
            prob_plot = E + (prob_norm * scale_factor / (abs(E-u_min * 1.2)))

            plt.axhline(y=E, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        
            plt.plot(x_ang, prob_plot, linewidth=1.5, label=r'$|\psi_{{{}}}|^2$'.format(i))
            
            plt.fill_between(x_ang, E, prob_plot, alpha=0.3)


        y_lim_sup = (np.max(E_phys) - np.min(U_phys)) * scaling_ylim_sup
        y_lim_inf = np.min(U_phys) * scaling_ylim_inf
        valid_indices = np.where(U_phys < np.max(y_lim_sup))[0]
        if len(valid_indices) > 0:
            plt.xlim(x_ang[valid_indices[0]], x_ang[valid_indices[-1]])
            plt.ylim(y_lim_inf, y_lim_sup) 

        plt.xlabel(r'Reaction Coordinate (Bohr amu$^{1/2}$)')
        plt.ylabel(y_label)
        plt.title(graph_title)
        
        if leg_loc:
            plt.legend(loc=leg_loc)
        else:
            plt.legend()
            
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if savefig_file:
            plt.savefig(savefig_file, dpi=300)
        plt.show()









    def plot_symmetric_prob_dens(self, 
                            n_levels=4, 
                            red_mass=1.0, 
                            scale_factor=0.005, 
                            units='Hartree',
                            savefig_file='',
                            scaling_ylim_sup = 0.4,
                            scaling_ylim_inf = 1.03,
                            graph_title = 'Symmetric DWP Probability Density',
                            leg_loc=''
                            ):
        """
        Plots the symmetric double well potential and probability densities (|psi|^2) using 
        a robust Finite Difference (grid) approach.
        """
        if not hasattr(self, 'h_vec') or len(self.h_vec) == 0:
            print("Error: No calculation results found.")
            return
        
        red_mass = red_mass * const.amu_to_au

        h = self.h_vec[-1]
        b = self.b_vec[-1]
        L = self.L_vec[-1]
        
        # grid - domain -pi/2 to pi/2
        N_grid = 2000
        epsilon = 1e-3
        y_grid = np.linspace(-np.pi/2 + epsilon, np.pi/2 - epsilon, N_grid)
        dy = y_grid[1] - y_grid[0]
        
        # [FIXED] reuse method for U(y), remove after testing
        # tan_y = np.tan(y_grid)
        # sin_y = np.sin(y_grid)
        # U_dim = h * (tan_y**2) + b * (sin_y**2)

        U_dim = self._aps_sym_potential(y_grid, h, b)
        U_dim = np.clip(U_dim, -1e9, 1e9)

        # build matrix
        kin_coeff = 1.0 / (dy**2)
        diag = 2 * kin_coeff + U_dim
        off_diag = -kin_coeff * np.ones(N_grid - 1)

        evals_dim, evecs = linalg.eigh_tridiagonal(
            diag, off_diag, select='i', select_range=(0, n_levels-1)
        )
        
        for i in range(len(evals_dim)):
            norm = np.sqrt(trapz(evecs[:, i]**2, y_grid))
            evecs[:, i] /= norm

        C_joules = (const.hbar_au**2 * np.pi**2) / (8 * red_mass * L**2)
        
        x_ang = (2 * L * y_grid / np.pi) #* 1e10 * const.bohr_to_ang  
        
        if units == 'kcal/mol':
            conv = 1.0 * const.hartree_to_kcalmol
            y_label = 'Energy (kcal/mol)'
        elif units == 'cm-1':
            conv = 1.0 * const.hartree_to_cm   
            y_label = 'Energy (cm$^{-1}$)'
        else:
            conv = 1.0 
            y_label = 'Energy (Hartree)'
            
        U_phys = U_dim * C_joules * conv
        E_phys = evals_dim * C_joules * conv

        plt.figure(figsize=(8, 6))
        
        e_max_plot = E_phys[-1]
        u_min = np.min(U_phys)
        
        # potential
        plt.plot(x_ang, U_phys, 'k-', linewidth=2, label='Potential')
        
        for i in range(len(E_phys)):
            E = E_phys[i]
            psi = evecs[:, i]
            prob_density = psi**2

            prob_norm = prob_density / np.max(prob_density)
            
            # [FIXED] slightly decreasing the norm as n_levels increases
            prob_plot = E + (prob_norm * scale_factor/ (abs(E-u_min * 1.2)))
                
            plt.axhline(y=E, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            
            plt.plot(x_ang, prob_plot, linewidth=1.5, label=r'$|\psi_{{{}}}|^2$'.format(i)) 
            plt.fill_between(x_ang, E, prob_plot, alpha=0.3)


        y_lim_inf = u_min * scaling_ylim_inf
        y_lim_sup = (np.max(E_phys) - u_min) * scaling_ylim_sup
        
        valid_indices = np.where(U_phys < np.max(E_phys) + y_lim_sup)[0]

        if len(valid_indices) > 0:
            plt.xlim(x_ang[valid_indices[0]], x_ang[valid_indices[-1]])
            plt.ylim(u_min * scaling_ylim_inf, e_max_plot + (e_max_plot - u_min)* scaling_ylim_sup)

        plt.xlabel(r'Reaction coordinate (Bohr amu$^{1/2}$)')
        plt.ylabel(y_label)
        plt.title(graph_title)
        
        if leg_loc:
            plt.legend(loc=leg_loc)
        else:
            plt.legend()
            
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if savefig_file:
            plt.savefig(savefig_file, dpi=300)
        plt.show()