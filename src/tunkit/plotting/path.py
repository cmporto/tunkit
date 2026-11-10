from matplotlib import pyplot as plt
import numpy as np
from tunkit.core import constants as const







class PathPlotMixin:

        # [TODO] so verbose... couldn't it be made better?
    def plot_curve(self, curve):
        '''Options for curve are 'raw_energies', 'projected', 'e_minus_v', 'sqrt', 'all'.  '''
        if curve == 'raw_energies' and self.raw_coord.size != 0 and self.energies.size != 0:
            plt.plot(np.linspace(0, len(self.raw_coord), len(self.energies)), 
                     (self.energies - self.energies[0]) * const.hartree_to_kcalmol)
            plt.scatter(np.linspace(0, len(self.raw_coord), len(self.energies)), 
                        (self.energies - self.energies[0]) * const.hartree_to_kcalmol,
                        c='C1', s=10)
        elif curve == 'projected' and self.proj_energies.size != 0:
            if len(self.reaction_path) == 0:
                plt.plot(np.linspace(0, len(self.raw_coord), len(self.energies)), 
                         (self.proj_energies - self.proj_energies[0]) * const.hartree_to_kcalmol)
                plt.scatter(np.linspace(0, len(self.raw_coord), len(self.energies)), 
                            (self.proj_energies - self.proj_energies[0]) * const.hartree_to_kcalmol,
                            c='C1', s=10)
            else:
                plt.plot(self.reaction_path, 
                         (self.proj_energies - self.proj_energies[0]) * const.hartree_to_kcalmol)
                plt.scatter(self.reaction_path, 
                            (self.proj_energies - self.proj_energies[0]) * const.hartree_to_kcalmol,
                            c='C1', s=10)
        elif curve == 'e_minus_v' and self.V_minus_E.size != 0:
            plt.scatter(self.reaction_path, self.V_minus_E * const.hartree_to_kcalmol,  c='C3', s=10)
            plt.plot(self.reaction_path, self.V_minus_E * const.hartree_to_kcalmol)
            plt.axhline(y=0.0, color='r')
        elif curve == 'sqrt' and self.sqroot.size != 0:
            plt.scatter(self.positive_rp, self.sqroot  * const.hartree_to_kcalmol,  c='C3', s=10)
            plt.plot(self.positive_rp, self.sqroot  * const.hartree_to_kcalmol)
            plt.axhline(y=0.0, color='r')
        elif curve == 'all' and \
            self.raw_coord.size != 0 and \
                self.energies.size != 0 and \
                    self.proj_energies.size != 0 and \
                        self.V_minus_E.size != 0 and self.sqroot.size != 0:
            fig, ax = plt.subplots(2,2)
            ax[0,0].plot(self.reaction_path, 
                     (self.energies - self.energies[0]) * const.hartree_to_kcalmol)
            ax[0,0].scatter(self.reaction_path, 
                        (self.energies - self.energies[0]) * const.hartree_to_kcalmol,
                        c='C3', s=10)
            ax[0,0].set_title("Electronic Energies")
            ax[0,1].plot(self.reaction_path, 
                         (self.proj_energies - self.proj_energies[0]) * const.hartree_to_kcalmol)
            ax[0,1].scatter(self.reaction_path, 
                            (self.proj_energies - self.proj_energies[0]) * const.hartree_to_kcalmol,
                            c='C3', s=10)
            ax[0,1].set_title("Proj. ZPVE Energies")
            ax[1,0].scatter(self.reaction_path, self.V_minus_E  * const.hartree_to_kcalmol,  c='C3', s=10)
            ax[1,0].plot(self.reaction_path, self.V_minus_E  * const.hartree_to_kcalmol)
            ax[1,0].axhline(y=0.0, color='r')
            ax[1,0].set_title("V(x) - E")
            ax[1,1].scatter(self.positive_rp, self.sqroot * const.hartree_to_kcalmol,  c='C3', s=10)
            ax[1,1].plot(self.positive_rp, self.sqroot  * const.hartree_to_kcalmol)
            ax[1,1].axhline(y=0.0, color='r')
            ax[1,1].set_title(r"$\sqrt{(V(x) - E)}$")
        else:
            print("Wrong energy name or you haven't loaded the coordinates and energies yet")
        plt.tight_layout()
        plt.show()








    def plot_path_diagnostics(self, 
                              arc_lengths, 
                              kinking, 
                              tangent_shifts, 
                              trust_radius=0.2, 
                              title_prefix="",
                              save_fig=''):
        """
        Generates a 3-panel diagnostic plot for NEB path health.
        """
        # Ensure we have data to plot
        if len(arc_lengths) == 0:
            print("No diagnostic data to plot.")
            return

        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        fig.suptitle(f'{title_prefix} Path Diagnostics', fontsize=14, fontweight='bold')

        # X-axes: Align indices properly based on array lengths
        x_arc = np.arange(1, len(arc_lengths) + 1)
        x_kink = np.arange(2, len(kinking) + 2)
        x_tan = np.arange(2, len(tangent_shifts) + 2)

        # graph 1: Arc lengths
        axs[0].plot(x_arc, arc_lengths, 'o-', color='royalblue')
        axs[0].axhline(y=trust_radius, color='red', linestyle='--', label=f'Trust Radius ({trust_radius})')
        axs[0].set_ylabel('Arc Length (Bohr/MW)')
        axs[0].set_title('Inter-Image Distance (Smaller is safer)')
        axs[0].legend()
        axs[0].grid(True, linestyle=':', alpha=0.7)

        # graph 2: kinking in path
        axs[1].plot(x_kink, kinking, 'o-', color='forestgreen')
        axs[1].axhline(y=0.99, color='red', linestyle='--', label='Warning Threshold (0.99)')
        axs[1].set_ylabel(r'Kink ($\cos \theta$)')
        axs[1].set_title('Path Smoothness (1.0 is perfectly straight)')
        axs[1].legend()
        axs[1].grid(True, linestyle=':', alpha=0.7)
        # Cap y axis to highlight the top margin where it should be
        axs[1].set_ylim(min(0.95, np.min(kinking) - 0.05), 1.01)

        # graph 3: Tangent consistency
        axs[2].plot(x_tan, tangent_shifts, 'o-', color='darkorange')
        axs[2].axhline(y=0.2, color='red', linestyle='--', label='Warning Threshold (0.2)')
        axs[2].set_xlabel('Image Index')
        axs[2].set_ylabel(r'$\|\Delta \hat{\tau}\|$')
        axs[2].set_title('Tangent Shift (Must be low for frequency projection)')
        axs[2].legend()
        axs[2].grid(True, linestyle=':', alpha=0.7)

        plt.tight_layout()
        if save_fig:
            plt.savefig(save_fig)
        plt.show()
