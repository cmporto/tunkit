import numpy as np
from tunkit.core import constants as const






def calc_center_of_mass(coords, atom_masses, verbose=False):
    masses = np.sum(atom_masses)
    com = np.array([0.0, 0.0, 0.0])
    for j, xyz in enumerate(coords):
        com = com + (xyz * atom_masses[j])
    if verbose:        
        print(masses)
    com = com / masses
    return com







def kabsch_alignment_cm(unalign, target, masses):

    centroid_unalign = calc_center_of_mass(unalign, masses)
    centroid_target = calc_center_of_mass(target, masses)

    unalign_centered = unalign - centroid_unalign
    target_centered = target - centroid_target


    # Multiply each atom's coordinates by its mass before the dot product
    # Reshaping self.masses to (N, 1) allows row-wise broadcasting
    masses_col = masses[:, np.newaxis] 
    H = unalign_centered.T @ (target_centered * masses_col) # H = sum(w_i x_i (y_i)^T)  where w_i = sqrt(w_i)*sqrt(w_i)
        
    # SVD and Rotation
    U, S, Vt = np.linalg.svd(H)
    Rot = Vt.T @ U.T
    
    # Correct for reflection
    if np.linalg.det(Rot) < 0:
        Vt[-1, :] *= -1
        Rot = Vt.T @ U.T

    # Calculate Translation and apply alignment
    Transl = centroid_target - Rot @ centroid_unalign
    unalign_aligned = (Rot @ unalign.T).T + Transl

    return unalign_aligned, Rot, Transl






def align_coordinates(coords, masses):  
    R_sqrt = []
    trans_vec = []
    aligned_coords = []
    un_center_of_mass = []
    rmsd = []
    for n, struct in enumerate(coords):
        # handling first frame
        if n == 0:
            aligned_coords.append(struct)
            struct2_aligned = struct
            un_center_of_mass.append(calc_center_of_mass(struct, masses))
            continue
        
        struct1 = struct2_aligned
        struct2 = coords[n]

        struct2_aligned, R_est, t_est = kabsch_alignment_cm(struct2, struct1, masses)

        R_sqrt.append(R_est)
        trans_vec.append(t_est)
        aligned_coords.append(struct2_aligned)
        un_center_of_mass.append(calc_center_of_mass(struct2_aligned, masses))
        
        # rmsd.append(np.sqrt(np.mean(np.sum((struct2_aligned - struct1)**2))))
        # [FIXED] rmsd calculation
        sq_diff = (struct2_aligned - struct1)**2
        
        # axis=1 to sum across coordinates, keeping one value per atom
        atom_sq_dists = np.sum(sq_diff, axis=1)
        

        step_rmsd = np.sqrt(np.mean(atom_sq_dists))
        rmsd.append(step_rmsd)
    
    center_of_mass = np.array(un_center_of_mass)
    rotational_matrices = np.array(R_sqrt)
    transl_vectors = np.array(trans_vec)
    alignement_rmsd = np.array(rmsd)        
    aligned_coord = np.array(aligned_coords)

    return center_of_mass, rotational_matrices, transl_vectors, alignement_rmsd, aligned_coord






def mass_weight_coordinates(coordinates, masses, verbose=False):
    """
    Generates mass-weighted coordinates using NumPy broadcasting.
    
    Expected Shapes:
    coordinates: (n_steps, n_atoms, 3)
    masses: (n_atoms,)
    """

    sqrt_masses = np.sqrt(masses)

    # shapes must be (n_atoms,) -> (1, n_atoms, 1)
    # this allows multiplying to (n_steps, n_atoms, 3)
    mass_factors = sqrt_masses[np.newaxis, :, np.newaxis]

    mass_scaled_coord = coordinates * mass_factors
    
    if verbose:
        print(f"Mass-weighted coordinates generated. Shape: {mass_scaled_coord.shape}")

    return mass_scaled_coord








def sqrt_eigval_convert_to_cm(vals):
    freqs = []
    for e in vals:
        sign = 1 if e >= 0 else -1 
        freqs.append(np.sqrt(abs(e)) * const.hartree_to_cm * sign)
    return freqs








