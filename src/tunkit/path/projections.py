import numpy as np
from dataclasses import replace
from tunkit.core import constants as const
from tunkit.core import geometry as geom
from tunkit.path import diagnostics 







def _project_out_tr(coords, target_mw, masses):
    """
    Internal helper: Removes Translational/Rotational components from a 
    Mass-Weighted Gradient (1D) or Hessian (2D).
    """
    n_atoms = len(masses)
    dim = 3 * n_atoms
    

    m_sqrt = np.sqrt(np.array(masses) * const.amu_to_au) 
    m_repeat = np.repeat(m_sqrt, 3) 
    

    D = np.zeros((dim, 6))
    
    # translations 0=2
    for i in range(3):
        D[i::3, i] = m_repeat[i::3]

    # rotations 3-5
    com = np.average(coords, axis=0, weights=masses)
    coords_shift = coords - com
    
    D[1::3, 3] = -coords_shift[:, 2] * m_repeat[1::3] 
    D[2::3, 3] =  coords_shift[:, 1] * m_repeat[2::3] 
    
    D[0::3, 4] =  coords_shift[:, 2] * m_repeat[0::3] 
    D[2::3, 4] = -coords_shift[:, 0] * m_repeat[2::3] 
    
    D[0::3, 5] = -coords_shift[:, 1] * m_repeat[0::3] 
    D[1::3, 5] =  coords_shift[:, 0] * m_repeat[1::3] 
    
    # orthonormalize
    Q, _ = np.linalg.qr(D)
    
    # project
    P = np.eye(dim) - Q @ Q.T
    
    if target_mw.ndim == 1:
        # gradient vector: P * g
        return P @ target_mw
    elif target_mw.ndim == 2:
        # hessian matrix: P.T * H * P
        return P.T @ target_mw @ P
    else:
        raise ValueError("target_mw must be 1D (gradient) or 2D (Hessian)")








def calculate_zpe(frequencies, cutoff=10.0):
    """
    Calculates the Zero Point Energy (ZPE) in Hartree units.
    
    Parameters:
    -----------
    frequencies : list or np.array
        List of frequencies in cm^-1 (output from project_frequencies).
    cutoff : float
        Threshold in cm^-1 to ignore projected modes (zeros) 
        or imaginary frequencies. Default is 10.0 cm^-1.
        
    Returns:
    --------
    zpe_hartree : float
        The ZPE in Hartree.
    """
    
    zpe_sum = 0.0
    
    for freq in frequencies:
        # sum 1/2 * h * v
        if freq > cutoff:
            zpe_sum += 0.5 * freq
            
    return zpe_sum * const.cm_to_hartree 








def project_frequencies(masses, coords, hessian, tangent
                        , projection_vector=None, threshold=1.0E-4):
    """
    Project TR + Gradient (7 modes) and return cm^-1 frequencies.
    """
    n_atoms = len(masses)
    dim = 3 * n_atoms
    
    # mass weighting the hessian and gradient
    m_repeat = np.repeat(masses, 3)
    mw_factor = np.sqrt(m_repeat * const.amu_to_au)
    m_root_inv = 1.0 / mw_factor 
    
    hessian_mw = hessian * np.outer(m_root_inv, m_root_inv)
    
    # tangent_mw = gradient * m_root_inv 
    # This was for mass weighting the gradient not the tangent.
    # Gradients are dE/dx so the mass weighting is in the denominator, dividing.
    # Tangents should be mass weighted like coordinates, multiplying
    tangent_mw = tangent * mw_factor
    
    # self.hessian_mw = hessian_mw
    # self.gradients_mw = tangent_mw

    # projection matrix 7 columns
    D = np.zeros((dim, 7))
    
    # translations are done in columns 0-2
    mw_atom_vec = np.sqrt(masses * const.amu_to_au) 
    for i in range(3):
        D[i::3, i] = mw_atom_vec
    
    # self.D = D

    # rotations are done in columns 3-5
    com = np.average(coords, axis=0, weights=masses)
    coords_shift = coords - com
    
    # Rx, Ry, Rz
    D[1::3, 3] = -coords_shift[:, 2] * mw_atom_vec 
    D[2::3, 3] =  coords_shift[:, 1] * mw_atom_vec 
    
    D[0::3, 4] =  coords_shift[:, 2] * mw_atom_vec 
    D[2::3, 4] = -coords_shift[:, 0] * mw_atom_vec 
    
    D[0::3, 5] = -coords_shift[:, 1] * mw_atom_vec 
    D[1::3, 5] =  coords_shift[:, 0] * mw_atom_vec 
    
    # self.D2 = D

    # mep vector (geometric gtradient)
    norm = np.linalg.norm(tangent_mw)
    if norm > 1e-14:
        D[:, 6] = tangent_mw / norm
    else:
        # if tangent is zero (duplicate points)
        print('WARNING!!! Tangent norm is below 1e-14! Probably, there are duplicate points! Check the graph!')
        # D[:, 6] = tangent_mw / np.linalg.norm(tangent_mw)

    Q, _ = np.linalg.qr(D)
    
    P = np.eye(dim) - Q @ Q.T
    H_proj = P.T @ hessian_mw @ P
    
    eigvals = np.linalg.eigvalsh(H_proj)

    # Changed into a utility method. Leaving it here to test, remove once everything is correct.
    # freqs = []
    # for e in eigvals:
    #     sign = 1 if e >= 0 else -1
    #     freqs.append(np.sqrt(abs(e)) * const.hartree_to_cm * sign)

    freqs = geom.sqrt_eigval_convert_to_cm(eigvals)

    return sorted(freqs), hessian_mw, tangent_mw








def process_path_data(proj_coordinates,
                      hessians,
                      masses,
                      energies, 
                      threshold=1.0E-4, 
                      diagnostic_path_freq=False, 
                      verbose=True):
    """
    Processes loaded path data (coordinates, energies, gradients, hessians) to calculate 
    tangents, mass-weight the matrices, project frequencies, and calculate ZPE.
    """
    if verbose:
        print("Processing path data: calculating tangents, projecting frequencies, and extracting normal modes...")

    n_steps = len(proj_coordinates)
    
    tangents = np.gradient(proj_coordinates, axis=0)

    correction = []
    mw_hessian_list = []
    mw_gradient_list = []
    path_freqs = []
    proj_freqs_list = []
    normal_mod = []

    gradient_rms = []

    for n in range(n_steps):
        tangent = tangents[n].flatten()

        proj_freqs, hessian_mw, gradient_mw = project_frequencies(
            masses, proj_coordinates[n], hessians[n], tangent, threshold=threshold
        )
        
        proj_freqs_list.append(proj_freqs)
        mw_hessian_list.append(hessian_mw)
        mw_gradient_list.append(gradient_mw)

        current_rms = np.sqrt(np.mean(gradient_mw**2))
        gradient_rms.append(current_rms)

        zpe = calculate_zpe(proj_freqs)
        correction.append(zpe)

        clean_h_mw = _project_out_tr(proj_coordinates[n], hessian_mw, masses)
        vals, vecs = np.linalg.eigh(clean_h_mw)
        
        # Changed into a utility method. Leaving it here to test, remove once everything is correct.
        # step_freqs = []
        # for e in vals:
        #     sign = 1 if e >= 0 else -1 #[111111]
        #     step_freqs.append(np.sqrt(abs(e)) * const.hartree_to_cm * sign)
            
        # path_freqs.append(step_freqs)

        path_freqs.append(geom.sqrt_eigval_convert_to_cm(vals))
        normal_mod.append(vecs)

    # 5. Store all arrays in class attributes
    projected_corrections = np.array(correction)
    mw_hessians = np.array(mw_hessian_list)
    mw_gradients = np.array(mw_gradient_list)
    normal_modes = np.array(normal_mod)
    path_frequencies = np.array(path_freqs)
    projected_frequencies = np.array(proj_freqs_list)
    
    proj_energies = np.array(energies) + projected_corrections
    
    if diagnostic_path_freq:
        diagnostics.diagnostic_path_and_proj_freq(n_steps, proj_freqs_list, path_frequencies)

    if len(proj_energies) > 0:
        bh = np.max(proj_energies) - proj_energies[0]
        kcal_val = bh * const.hartree_to_kcalmol
        cm_val = bh * const.hartree_to_cm
        if verbose:
            print('Vª barrier height is {:.2f} kcal/mol or {:.2f} cm-1.'.format(kcal_val, cm_val))

    return {'projected_corrections': projected_corrections,
           'mw_hessians':mw_hessians,
           'normal_modes':normal_modes,
           'mw_gradients':mw_gradients,
           'gradient_rms':gradient_rms,
           'path_frequencies':path_frequencies,
           'tangents':tangents,
           'projected_frequencies':projected_frequencies,
           'proj_energies':proj_energies}











# class ProjectionMixin:

