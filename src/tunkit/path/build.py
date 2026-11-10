import numpy as np
from tunkit.core import geometry as geom
from dataclasses import replace
# from tunkit.path.reaction_path import ReactionPath
# from matplotlib import pyplot as plt
# from dataclasses import fields, replace









def build_reaction_path(path=None, align_raw=False, align_mw=True, verbose=False):
    # align the coordinates
    # if align_raw:
    #     geom.align_coordinates(self.raw_coord, self.masses, mass_weighted=False)
    #     print("Coordinates have been aligned!")
    if path.proj_coordinates is not None: # or len(path.proj_coordinates) > 0: # None type has no len
        center_of_mass, rotational_matrices, transl_vectors, \
        alignement_rmsd, aligned_coord = geom.align_coordinates(path.proj_coordinates, path.masses)
    else:
        center_of_mass, rotational_matrices, transl_vectors, \
        alignement_rmsd, aligned_coord = geom.align_coordinates(path.raw_coord, path.masses)
        if verbose:
            print("Using non-projected coordinates, mind you...")
    if verbose:
        print("Coordinates have been aligned!")
    mass_scaled_coord = geom.mass_weight_coordinates(aligned_coord, path.masses)
    # if len(self.mass_scaled_coord) == 0:
    #     if len(self.aligned_coord) == 0:
    #         self.mass_weight_coordinates(self.raw_coord, verbose=verbose)
    #     else:
    #         self.mass_weight_coordinates(self.aligned_coord, verbose=verbose)
    # else:
    #     print('WARNING: Mass scaled coordinates already built... using it')
    # if align_mw:
    #     geom.align_coordinates(self.mass_scaled_coord, self.masses, mass_weighted=mass_weighted)
    #     print("Mass weighted coordinates have been aligned!")
    #     self.mass_scaled_coord = self.aligned_coord 
    displacement_per_step = build_ds(mass_scaled_coord)
    displacement_non_mw = build_ds(path.raw_coord)

    reaction_path = sum_ds_to_path(path, displacement_per_step, verbose=verbose)
    cartesian_disp_non_mw = sum_ds_to_path(path, displacement_non_mw, verbose=verbose)  

    if verbose:
        print("Reaction path built!")

    path = replace(path,
                   center_of_mass=center_of_mass, 
                   rotational_matrices=rotational_matrices, 
                   transl_vectors=transl_vectors, 
                   alignement_rmsd=alignement_rmsd, 
                   mass_scaled_coord=mass_scaled_coord, 
                   displacement_per_step=displacement_per_step,
                   displacement_non_mw=displacement_non_mw,
                   reaction_path=reaction_path,
                   cartesian_disp_non_mw=cartesian_disp_non_mw)

    return path

    # self.path = self.to_path()
    # return self.path








def build_ds(coordinates, verbose=True):
    """
    Calculates the scalar displacement (arc length) between consecutive steps.
    
    Expected Input:
    coordinates: (n_steps, n_atoms, 3) 
                *Should be the mass-weighted coordinates from the previous method*
    """

    # automatically handles the n=0 
    # shape (n_steps - 1, n_atoms, 3)
    deltas = np.diff(coordinates, axis=0)

    deltas_sq = deltas ** 2

    sum_sq = np.sum(deltas_sq, axis=(1, 2))

    # sqrt at the end dummy
    return np.sqrt(sum_sq)








def sum_ds_to_path(path, displacement_per_step, verbose=True):    
    if path.energies is None or len(path.energies) == 0:
        print("Error: Energies not loaded. Please run extract_projection_energies_neb() or extract_projection_energies_irc() first.")
        return
    # [FIXED] We are using the raw energies and not the projected energies. Should do a if clause that if we have the proj energies then use them.
    if path.proj_energies is not None: # len(path.proj_energies) > 0: # None type has no len
        index_top = path.proj_energies.argmax()
    else:
        index_top = path.energies.argmax() # [TODO] Warning about using non projected energies should not be silent!!
    path1 = displacement_per_step[:index_top][::-1]
    path2 = displacement_per_step[index_top:]
    reaction_path1 = []
    cumulative_ds = 0
    for ds in path1:
        cumulative_ds -= ds
        reaction_path1.append(cumulative_ds)
    reaction_path2 = []
    cumulative_ds = 0
    for ds in path2:
        cumulative_ds += ds
        reaction_path2.append(cumulative_ds)
    path = reaction_path1[::-1] + [0.0] + reaction_path2
    return np.array([x for x in path]) #* 1e-10 








# class PathBuildingMixin:

    # def build_ds(self, coordinates, verbose=True):
    #     """
    #     Calculates the scalar displacement (arc length) between consecutive steps.
        
    #     Expected Input:
    #     coordinates: (n_steps, n_atoms, 3) 
    #                 *Should be the mass-weighted coordinates from the previous method*
    #     """

    #     # automatically handles the n=0 
    #     # shape (n_steps - 1, n_atoms, 3)
    #     deltas = np.diff(coordinates, axis=0)

    #     deltas_sq = deltas ** 2

    #     sum_sq = np.sum(deltas_sq, axis=(1, 2))

    #     # sqrt at the end dummy
    #     return np.sqrt(sum_sq)








    # def sum_ds_to_path(self, displacement_per_step, verbose=True):    
    #     if self.energies is None or len(self.energies) == 0:
    #         print("Error: Energies not loaded. Please run extract_projection_energies_neb() or extract_projection_energies_irc() first.")
    #         return
    #     # [FIXED] We are using the raw energies and not the projected energies. Should do a if clause that if we have the proj energies then use them.
    #     if len(self.proj_energies) > 0:
    #         index_top = self.proj_energies.argmax()
    #     else:
    #         index_top = self.energies.argmax()
    #     path1 = displacement_per_step[:index_top][::-1]
    #     path2 = displacement_per_step[index_top:]
    #     reaction_path1 = []
    #     cumulative_ds = 0
    #     for ds in path1:
    #         cumulative_ds -= ds
    #         reaction_path1.append(cumulative_ds)
    #     reaction_path2 = []
    #     cumulative_ds = 0
    #     for ds in path2:
    #         cumulative_ds += ds
    #         reaction_path2.append(cumulative_ds)
    #     path = reaction_path1[::-1] + [0.0] + reaction_path2
    #     return np.array([x for x in path]) #* 1e-10 








    # def build_reaction_path(self, align_raw=False, align_mw=True, verbose=False):
    #     print('old build_reaction_path')
    #     # align the coordinates
    #     # if align_raw:
    #     #     geom.align_coordinates(self.raw_coord, self.masses, mass_weighted=False)
    #     #     print("Coordinates have been aligned!")
    #     if len(self.proj_coordinates) > 0:
    #         self.center_of_mass, self.rotational_matrices, self.transl_vectors, self.alignement_rmsd, self.aligned_coord = geom.align_coordinates(self.proj_coordinates, self.masses)
    #     else:
    #         self.center_of_mass, self.rotational_matrices, self.transl_vectors, self.alignement_rmsd, self.aligned_coord = geom.align_coordinates(self.raw_coord, self.masses)
    #         if verbose:
    #             print("Using non-projected coordinates, mind you...")
    #     if verbose:
    #         print("Coordinates have been aligned!")
    #     self.mass_scaled_coord = geom.mass_weight_coordinates(self.aligned_coord, self.masses)
    #     # if len(self.mass_scaled_coord) == 0:
    #     #     if len(self.aligned_coord) == 0:
    #     #         self.mass_weight_coordinates(self.raw_coord, verbose=verbose)
    #     #     else:
    #     #         self.mass_weight_coordinates(self.aligned_coord, verbose=verbose)
    #     # else:
    #     #     print('WARNING: Mass scaled coordinates already built... using it')
    #     # if align_mw:
    #     #     geom.align_coordinates(self.mass_scaled_coord, self.masses, mass_weighted=mass_weighted)
    #     #     print("Mass weighted coordinates have been aligned!")
    #     #     self.mass_scaled_coord = self.aligned_coord 
    #     self.displacement_per_step = self.build_ds(self.mass_scaled_coord)
    #     self.displacement_non_mw = self.build_ds(self.raw_coord)

    #     self.reaction_path = self.sum_ds_to_path(self.displacement_per_step, verbose=verbose)
    #     self.cartesian_disp_non_mw = self.sum_ds_to_path(self.displacement_non_mw, verbose=verbose)  

    #     if verbose:
    #         print("Reaction path built!")

    #     self.path = self.to_path()
    #     return self.path









    # def to_path(self):
    #     kwargs = {}
    #     for f in fields(ReactionPath):
    #         if not hasattr(self, f.name):
    #             continue
    #         val = getattr(self, f.name)
    #         if f.name == 'direction':
    #             kwargs[f.name] = val
    #         else:
    #             kwargs[f.name] = np.asarray(val)
    #     return ReactionPath(**kwargs)


    # # def to_path(self):
    # #     """Collect the current state into a ReactionPath."""
    # #     return ReactionPath(**{f.name: getattr(self, f.name)
    # #                         for f in fields(ReactionPath)
    # #                         if hasattr(self, f.name)})

    # def from_path(self, path):
    #     """Write a ReactionPath back onto self."""
    #     for f in fields(ReactionPath):
    #         setattr(self, f.name, getattr(path, f.name))

    # def _reverse_path_data(self):
    #     # print('new')
    #     self.from_path(self.to_path().reversed())

    # def clip_reaction_path(self, clip_start=0, clip_end=0, plot_curve=True, verbose=True):
    #     # print('new')
    #     self.from_path(self.to_path().clipped(clip_start, clip_end))

    #     if verbose:
    #         print(f"Reaction path clipped: {clip_start} steps from start, {clip_end} steps from end.")
    #         print(f"New path length: {len(self.reaction_path)} steps.")
        
    #     if plot_curve:
    #         plt.figure(figsize=(8, 5))
    #         plt.plot(self.reaction_path, self.proj_energies, marker='o')
    #         plt.title('Clipped Reaction Path')
    #         plt.xlabel(r'Reaction coordinates (Bohr.amu$^{1/2}$)')
    #         plt.ylabel('Energy (Hartree)')
    #         plt.grid()
    #         plt.show()








    # # [TODO] theoretically instead of using a bunch of class variables, I should use some
    # # kind of data class, dictionary or data struct.
    # def clip_reaction_path2(self, clip_start=0, clip_end=0, plot_curve=True, verbose=True):
    #     """
    #     Clips the reaction path and associated data arrays to focus on a specific region.
        
    #     Parameters:
    #     -----------
    #     clip_start : int
    #         Number of steps to clip from the start (reactant side).
    #     clip_end : int
    #         Number of steps to clip from the end (product side).
    #     verbose : bool
    #         If True, prints information about the clipping process.
    #     """
    #     if len(self.reaction_path) == 0:
    #         print("Error: Reaction path not built yet. Please run build_reaction_path() first.")
    #         return
        
    #     # new indices after clipping
    #     start_idx = clip_start
    #     end_idx = len(self.reaction_path) - clip_end
        
    #     if start_idx >= end_idx:
    #         print("Error: Clip values are too large, resulting in an empty path.")
    #         return
        
    #     # [IMPORTANT] if a new one is added, remember to add it here
    #     # things can go very wrong if it's not added here [TODO] ain't no better way??? Dictionaries maybe?
    #     self.reaction_path = self.reaction_path[start_idx:end_idx].copy() # Use copy to avoid referencing the original array
    #     self.proj_energies = self.proj_energies[start_idx:end_idx].copy()
    #     self.projected_corrections = self.projected_corrections[start_idx:end_idx].copy()
    #     self.hessians = self.hessians[start_idx:end_idx].copy()
    #     self.gradients = self.gradients[start_idx:end_idx].copy()
    #     self.proj_coordinates = self.proj_coordinates[start_idx:end_idx].copy()
    #     self.energies = self.energies[start_idx:end_idx].copy()
    #     self.mw_hessians = self.mw_hessians[start_idx:end_idx].copy()
    #     self.mw_gradients = self.mw_gradients[start_idx:end_idx].copy()
    #     self.normal_modes = self.normal_modes[start_idx:end_idx].copy()
    #     self.path_frequencies = self.path_frequencies[start_idx:end_idx].copy()
    #     self.mass_scaled_coord = self.mass_scaled_coord[start_idx:end_idx].copy()
    #     self.aligned_coord = self.aligned_coord[start_idx:end_idx].copy()
    #     self.tangents = self.tangents[start_idx:end_idx].copy()
    #     self.projected_frequencies = self.projected_frequencies[start_idx:end_idx].copy()
    #     self.gradient_rms = self.gradient_rms[start_idx:end_idx].copy()
    #     self.raw_coord = self.raw_coord[start_idx:end_idx].copy()
    #     self.raw_coord_ang = self.raw_coord_ang[start_idx:end_idx].copy()
    #     self.cartesian_disp_non_mw = self.cartesian_disp_non_mw[start_idx:end_idx].copy()

        
    #     if verbose:
    #         print(f"Reaction path clipped: {clip_start} steps from start, {clip_end} steps from end.")
    #         print(f"New path length: {len(self.reaction_path)} steps.")
        
    #     if plot_curve:
    #         plt.figure(figsize=(8, 5))
    #         plt.plot(self.reaction_path, self.proj_energies, marker='o')
    #         plt.title('Clipped Reaction Path')
    #         plt.xlabel(r'Reaction coordinates (Bohr.amu$^{1/2}$)')
    #         plt.ylabel('Energy (Hartree)')
    #         plt.grid()
    #         plt.show()








    # def _reverse_path_data2(self):
    #     """
    #     Helper to reverse all trajectory data arrays (Reactant <-> Product).
    #     Used to re-use reactant-based logic for product analysis.
    #     """
    #     if len(self.aligned_coord) > 0: self.aligned_coord = self.aligned_coord[::-1]
    #     if len(self.raw_coord) > 0: self.raw_coord = self.raw_coord[::-1]
    #     if len(self.energies) > 0: self.energies = self.energies[::-1]
    #     if len(self.proj_energies) > 0: self.proj_energies = self.proj_energies[::-1]
    #     if len(self.projected_corrections) > 0: self.projected_corrections = self.projected_corrections[::-1]
    #     if len(self.mass_scaled_coord) > 0: self.mass_scaled_coord = self.mass_scaled_coord[::-1]
    #     if len(self.reaction_path) > 0: 
    #         # Reaction path needs sign flip AND reversal for correct coordinate frame
    #         # [TODO] check if only reversal without sign flip works. Possible bug!!    
    #         # Changed to sign reversal
    #         self.reaction_path = -self.reaction_path[::-1]
    #     if len(self.tangents) > 0: self.tangents = self.tangents[::-1]
    #     if len(self.projected_frequencies) > 0: self.projected_frequencies = self.projected_frequencies[::-1]
    #     if len(self.gradient_rms) > 0: self.gradient_rms = self.gradient_rms[::-1]  

    #     if len(self.proj_coordinates) > 0: self.proj_coordinates = self.proj_coordinates[::-1]
    #     if len(self.gradients) > 0: self.gradients = self.gradients[::-1]
    #     if len(self.mw_gradients) > 0: self.mw_gradients = self.mw_gradients[::-1]
    #     if len(self.hessians) > 0: self.hessians = self.hessians[::-1]
    #     if len(self.mw_hessians) > 0: self.mw_hessians = self.mw_hessians[::-1]
    #     if len(self.normal_modes) > 0: self.normal_modes = self.normal_modes[::-1]
    #     if len(self.path_frequencies) > 0: self.path_frequencies = self.path_frequencies[::-1]







