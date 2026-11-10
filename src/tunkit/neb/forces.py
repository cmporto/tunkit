import numpy as np
from tunkit.core import constants as const
from tunkit.core import elements as elem
from tunkit.core import geometry as geom
from tunkit.path import projections 








def _neb_compute_forces(images_mw, 
                        energies, 
                        forces_mw,
                        k_spring, 
                        masses,
                        frozen_indices=None):
    """
    Compute NEB forces for all movable images and convergence metrics.

    Always returns perpendicular and spring forces separately.
    The user decides whether to combine or separate them for the
    optimizer.

    T+R projection is applied to every interior image's true force
    in MW space before the parallel/perpendicular split.

    Convergence metrics use the perpendicular true force in Cartesian
    space (Eh/bohr), matching ORCA's tol_maxfp_i / tol_rmsfp_i.

    Parameters
    ----------
    images_mw : list of ndarray (N_atoms, 3)
    energies  : ndarray (N_images,)
    forces_mw : list of ndarray (N_atoms, 3)
    k_spring  : float
    frozen_indices : set of int, optional

    Returns
    -------
    neb_forces_perp   : dict {int: ndarray (N_atoms, 3)}
    neb_forces_spring : dict {int: ndarray (N_atoms, 3)}
    fmax : float - max |F_perp_cart| over movable images
    frms : float - rms  F_perp_cart  over movable images
    """

    n_images = len(images_mw)
    n_atoms  = len(masses)
    sqrt_m   = np.sqrt(masses)[:, np.newaxis]

    if frozen_indices is None:
        frozen_indices = {0, n_images - 1}

    neb_forces_perp   = {}
    neb_forces_spring = {}
    all_fperp_cart    = []
    fmax_per_image = {}

    # Diagnostic tracking variables
    theta_max  = 0.0
    df_err_max = 0.0

    for i in range(1, n_images - 1):
        # Project T+R out of the true MW force
        f_raw_flat   = forces_mw[i].flatten()
        # [NOALIGN] as part of the effort to avoid rotations
        # we are only projecting the forces at the end
        # f_clean_flat = projections._project_out_tr(images_mw[i], f_raw_flat)

        # Tangent
        tau      = _neb_compute_tangent(energies, images_mw, i)
        tau_flat = tau.flatten()

        # Parallel / perpendicular split
        f_par_component = np.dot(f_raw_flat, tau_flat)
        f_parallel      = f_par_component * tau_flat
        f_perp_mw       = f_raw_flat - f_parallel

        # Skip frozen images for optimizer + convergence
        if i in frozen_indices:
            continue


        # Spring force (parallel to tangent)
        d_plus = np.linalg.norm(
            (images_mw[i + 1] - images_mw[i]).flatten()
        )
        d_minus = np.linalg.norm(
            (images_mw[i] - images_mw[i - 1]).flatten()
        )
        f_spring_parallel = k_spring * (d_plus - d_minus) * tau_flat

        # Passing Cartesian coordinates to the fixed _project_out_tr method
        cart_coords = images_mw[i] / sqrt_m
        f_perp_clean   = projections._project_out_tr(cart_coords, f_perp_mw, masses)
        f_spring_clean = projections._project_out_tr(cart_coords, f_spring_parallel, masses)


        # Rotational drift (theta in degrees) relative to previous image
        # since theta is the exp of WKB, maybe change the greek letters...
        cart_ref = images_mw[i - 1] / sqrt_m
        _, Rot, _ = geom.kabsch_alignment_cm(cart_coords, cart_ref, masses)
        trace_R = np.trace(Rot)
        val = np.clip((trace_R - 1.0) / 2.0, -1.0, 1.0)
        drift_deg = float(np.degrees(np.arccos(val)))
        theta_max = max(theta_max, drift_deg)

        # Projection Error (magnitude of T+R force removed from F_perp)
        f_diff_mw = f_perp_mw - f_perp_clean
        f_diff_cart = (f_diff_mw.reshape(-1, 3) * sqrt_m).flatten()
        err_mag = float(np.max(np.abs(f_diff_cart)))
        df_err_max = max(df_err_max, err_mag)

        # Convergence metric is the perpendicular force in Cartesian
        f_perp_cart = (f_perp_clean.reshape(-1, 3) * sqrt_m).flatten()
        all_fperp_cart.append(f_perp_cart)

        #Save the per image max force (debug)
        fmax_per_image[i] = float(np.max(np.abs(f_perp_cart)))

        neb_forces_perp[i]   = f_perp_clean.reshape(n_atoms, 3)
        neb_forces_spring[i] = f_spring_clean.reshape(n_atoms, 3)

    if not all_fperp_cart:
            return (
    neb_forces_perp,
    neb_forces_spring,
    0.0,
    0.0,
    theta_max,
    df_err_max,
    fmax_per_image,
)

    all_fperp = np.concatenate(all_fperp_cart)
    fmax = float(np.max(np.abs(all_fperp)))
    frms = float(np.sqrt(np.mean(all_fperp ** 2)))

    return neb_forces_perp, neb_forces_spring, fmax, frms, theta_max, df_err_max, fmax_per_image








def _neb_compute_tangent(energies, images_mw, idx):
    """
    Path tangent at image idx using the improved Henkelman-Jónsson 
    method (JCP 113, 9978, 2000).
    
    Uses only the higher-energy neighbor when the energy is monotonic, 
    preventing kink formation near the TS. At local extrema, uses an 
    energy-weighted average of both neighbors.
    
    Parameters
    ----------
    energies : ndarray, shape (N_images,)
        Energy of each image (Hartree).
    images_mw : list of ndarray, each (N_atoms, 3)
        Images in mass-weighted coordinates.
    idx : int
        Image index. Must be an interior image (0 < idx < N-1).
        
    Returns
    -------
    tau : ndarray, shape (N_atoms, 3)
        Normalized tangent vector in mass-weighted space.
    """
    n = len(images_mw)
    
    # Endpoints: simple forward/backward difference
    if idx == 0:
        tau = images_mw[1] - images_mw[0]
    elif idx == n - 1:
        tau = images_mw[-1] - images_mw[-2]
    else:
        tau_plus  = images_mw[idx + 1] - images_mw[idx]
        tau_minus = images_mw[idx] - images_mw[idx - 1]
        
        E_i   = energies[idx]
        E_ip1 = energies[idx + 1]
        E_im1 = energies[idx - 1]
        
        # Far from the barrier top
        if E_ip1 > E_i > E_im1:
            tau = tau_plus.copy()
        elif E_ip1 < E_i < E_im1:
            tau = tau_minus.copy()
        else:
            # Local extrema, do energy-weighted average
            dE_large = max(abs(E_ip1 - E_i), abs(E_i - E_im1))
            dE_small = min(abs(E_ip1 - E_i), abs(E_i - E_im1))
            
            if E_ip1 > E_im1:
                tau = dE_large * tau_plus + dE_small * tau_minus
            else:
                tau = dE_small * tau_plus + dE_large * tau_minus
    
    # Normalize
    norm = np.linalg.norm(tau)
    if norm > 1e-14:
        tau = tau / norm
    else:
        print(f"  WARNING: Near-zero tangent at image {idx}. Maybe duplicate points?")
    
    return tau







# class ForcesNebMixin:

    # def _neb_compute_forces(self, images_mw, energies, forces_mw,
    #                         k_spring, frozen_indices=None):
    #     """
    #     Compute NEB forces for all movable images and convergence metrics.
    
    #     Always returns perpendicular and spring forces separately.
    #     The user decides whether to combine or separate them for the
    #     optimizer.
    
    #     T+R projection is applied to every interior image's true force
    #     in MW space before the parallel/perpendicular split.
    
    #     Convergence metrics use the perpendicular true force in Cartesian
    #     space (Eh/bohr), matching ORCA's tol_maxfp_i / tol_rmsfp_i.
    
    #     Parameters
    #     ----------
    #     images_mw : list of ndarray (N_atoms, 3)
    #     energies  : ndarray (N_images,)
    #     forces_mw : list of ndarray (N_atoms, 3)
    #     k_spring  : float
    #     frozen_indices : set of int, optional
    
    #     Returns
    #     -------
    #     neb_forces_perp   : dict {int: ndarray (N_atoms, 3)}
    #     neb_forces_spring : dict {int: ndarray (N_atoms, 3)}
    #     fmax : float - max |F_perp_cart| over movable images
    #     frms : float - rms  F_perp_cart  over movable images
    #     """
    
    #     n_images = len(images_mw)
    #     n_atoms  = len(self.masses)
    #     sqrt_m   = np.sqrt(self.masses)[:, np.newaxis]
    
    #     if frozen_indices is None:
    #         frozen_indices = {0, n_images - 1}
    
    #     neb_forces_perp   = {}
    #     neb_forces_spring = {}
    #     all_fperp_cart    = []
    #     fmax_per_image = {}

    #     # Diagnostic tracking variables
    #     theta_max  = 0.0
    #     df_err_max = 0.0
    
    #     for i in range(1, n_images - 1):
    #         # Project T+R out of the true MW force
    #         f_raw_flat   = forces_mw[i].flatten()
    #         # [NOALIGN] as part of the effort to avoid rotations
    #         # we are only projecting the forces at the end
    #         # f_clean_flat = projections._project_out_tr(images_mw[i], f_raw_flat)
    
    #         # Tangent
    #         tau      = self._neb_compute_tangent(energies, images_mw, i)
    #         tau_flat = tau.flatten()
    
    #         # Parallel / perpendicular split
    #         f_par_component = np.dot(f_raw_flat, tau_flat)
    #         f_parallel      = f_par_component * tau_flat
    #         f_perp_mw       = f_raw_flat - f_parallel
    
    #         # Skip frozen images for optimizer + convergence
    #         if i in frozen_indices:
    #             continue
    

    #         # Spring force (parallel to tangent)
    #         d_plus = np.linalg.norm(
    #             (images_mw[i + 1] - images_mw[i]).flatten()
    #         )
    #         d_minus = np.linalg.norm(
    #             (images_mw[i] - images_mw[i - 1]).flatten()
    #         )
    #         f_spring_parallel = k_spring * (d_plus - d_minus) * tau_flat

    #         # Passing Cartesian coordinates to the fixed _project_out_tr method
    #         cart_coords = images_mw[i] / sqrt_m
    #         f_perp_clean   = projections._project_out_tr(cart_coords, f_perp_mw, self.masses)
    #         f_spring_clean = projections._project_out_tr(cart_coords, f_spring_parallel, self.masses)


    #         # Rotational drift (theta in degrees) relative to previous image
    #         # since theta is the exp of WKB, maybe change the greek letters...
    #         cart_ref = images_mw[i - 1] / sqrt_m
    #         _, Rot, _ = geom.kabsch_alignment_cm(cart_coords, cart_ref, self.masses)
    #         trace_R = np.trace(Rot)
    #         val = np.clip((trace_R - 1.0) / 2.0, -1.0, 1.0)
    #         drift_deg = float(np.degrees(np.arccos(val)))
    #         theta_max = max(theta_max, drift_deg)

    #         # Projection Error (magnitude of T+R force removed from F_perp)
    #         f_diff_mw = f_perp_mw - f_perp_clean
    #         f_diff_cart = (f_diff_mw.reshape(-1, 3) * sqrt_m).flatten()
    #         err_mag = float(np.max(np.abs(f_diff_cart)))
    #         df_err_max = max(df_err_max, err_mag)

    #         # Convergence metric is the perpendicular force in Cartesian
    #         f_perp_cart = (f_perp_clean.reshape(-1, 3) * sqrt_m).flatten()
    #         all_fperp_cart.append(f_perp_cart)

    #         #Save the per image max force (debug)
    #         fmax_per_image[i] = float(np.max(np.abs(f_perp_cart)))
    
    #         neb_forces_perp[i]   = f_perp_clean.reshape(n_atoms, 3)
    #         neb_forces_spring[i] = f_spring_clean.reshape(n_atoms, 3)
    
    #     if not all_fperp_cart:
    #             return (
    #     neb_forces_perp,
    #     neb_forces_spring,
    #     0.0,
    #     0.0,
    #     theta_max,
    #     df_err_max,
    #     fmax_per_image,
    # )
    
    #     all_fperp = np.concatenate(all_fperp_cart)
    #     fmax = float(np.max(np.abs(all_fperp)))
    #     frms = float(np.sqrt(np.mean(all_fperp ** 2)))
    
    #     return neb_forces_perp, neb_forces_spring, fmax, frms, theta_max, df_err_max, fmax_per_image








    # def _neb_compute_tangent(self, energies, images_mw, idx):
    #     """
    #     Path tangent at image idx using the improved Henkelman-Jónsson 
    #     method (JCP 113, 9978, 2000).
        
    #     Uses only the higher-energy neighbor when the energy is monotonic, 
    #     preventing kink formation near the TS. At local extrema, uses an 
    #     energy-weighted average of both neighbors.
        
    #     Parameters
    #     ----------
    #     energies : ndarray, shape (N_images,)
    #         Energy of each image (Hartree).
    #     images_mw : list of ndarray, each (N_atoms, 3)
    #         Images in mass-weighted coordinates.
    #     idx : int
    #         Image index. Must be an interior image (0 < idx < N-1).
            
    #     Returns
    #     -------
    #     tau : ndarray, shape (N_atoms, 3)
    #         Normalized tangent vector in mass-weighted space.
    #     """
    #     n = len(images_mw)
        
    #     # Endpoints: simple forward/backward difference
    #     if idx == 0:
    #         tau = images_mw[1] - images_mw[0]
    #     elif idx == n - 1:
    #         tau = images_mw[-1] - images_mw[-2]
    #     else:
    #         tau_plus  = images_mw[idx + 1] - images_mw[idx]
    #         tau_minus = images_mw[idx] - images_mw[idx - 1]
            
    #         E_i   = energies[idx]
    #         E_ip1 = energies[idx + 1]
    #         E_im1 = energies[idx - 1]
            
    #         # Far from the barrier top
    #         if E_ip1 > E_i > E_im1:
    #             tau = tau_plus.copy()
    #         elif E_ip1 < E_i < E_im1:
    #             tau = tau_minus.copy()
    #         else:
    #             # Local extrema, do energy-weighted average
    #             dE_large = max(abs(E_ip1 - E_i), abs(E_i - E_im1))
    #             dE_small = min(abs(E_ip1 - E_i), abs(E_i - E_im1))
                
    #             if E_ip1 > E_im1:
    #                 tau = dE_large * tau_plus + dE_small * tau_minus
    #             else:
    #                 tau = dE_small * tau_plus + dE_large * tau_minus
        
    #     # Normalize
    #     norm = np.linalg.norm(tau)
    #     if norm > 1e-14:
    #         tau = tau / norm
    #     else:
    #         print(f"  WARNING: Near-zero tangent at image {idx}. Maybe duplicate points?")
        
    #     return tau