import numpy as np
# import os
from scipy.interpolate import CubicSpline
# from tunkit.core import elements as elem
from tunkit.core import geometry as geom
# from tunkit.core import engine
# from tunkit.neb import neb_io








def _neb_align_band(images_mw, masses, frozen_indices=None, opt_state=None, scope='global'):
    """Chain-align each movable image onto its left neighbour (mass-weighted,
    via Cartesian), in place. Frozen images are anchors. Idempotent.
    Applies the rotation matrix to optimizer velocities and L-BFGS history."""
    sqrt_m = np.sqrt(masses)[:, np.newaxis]

    
    
    # If global scope, the velocities are stored as a single flat array
    if scope == 'global' and opt_state and 'v' in opt_state:
        n_atoms = len(masses)
        flat_v = opt_state['v'].copy()
        movable_idx = 0 
        
    for i in range(1, len(images_mw)):
        # SKIP FROZEN IMAGES to prevent anchor drift!
        # if i in frozen_indices:
        #     continue
            
        cart_i   = images_mw[i]     / sqrt_m
        cart_ref = images_mw[i - 1] / sqrt_m
        
        # ALIGN
        cart_aligned, _R, _T = geom.kabsch_alignment_cm(cart_i, cart_ref, masses)
        images_mw[i] = cart_aligned * sqrt_m
        
        # --- APPLY ROTATION TO OPTIMIZER STATE ---
        # [TODO] There are problems here (avoided by projecting the tr+rot on ALL
        #        forces and not aligning...)
        if opt_state:
            if scope == 'local' and i in opt_state:
                
                # A. VPO / FIRE Velocity (Vectors: Rotate only)
                if 'v' in opt_state[i]:
                    v_shaped = opt_state[i]['v'].reshape(-1, 3)
                    opt_state[i]['v'] = (v_shaped @ _R.T).flatten()
                    
                # B. L-BFGS History
                if 's_hist' in opt_state[i]:
                    # History vectors (s, y, prev_g): Rotate only (commutes with mass-weighting)
                    opt_state[i]['s_hist'] = [
                        (s.reshape(-1, 3) @ _R.T).flatten() for s in opt_state[i]['s_hist']
                    ]
                    opt_state[i]['y_hist'] = [
                        (y.reshape(-1, 3) @ _R.T).flatten() for y in opt_state[i]['y_hist']
                    ]
                    if 'prev_g' in opt_state[i] and opt_state[i]['prev_g'] is not None:
                        opt_state[i]['prev_g'] = (opt_state[i]['prev_g'].reshape(-1, 3) @ _R.T).flatten()
                    
                    # Previous coordinate array (prev_q): Convert to Cart -> Rotate & Translate -> Convert to MW
                    if 'prev_q' in opt_state[i] and opt_state[i]['prev_q'] is not None:
                        cart_prev = opt_state[i]['prev_q'].reshape(-1, 3) / sqrt_m
                        cart_prev_aligned = (cart_prev @ _R.T) + _T
                        opt_state[i]['prev_q'] = (cart_prev_aligned * sqrt_m).flatten()
                
                # C. future optimizers here should be added here too [TODO
                
            elif scope == 'global' and 'v' in opt_state:
                # Global scope VPO/FIRE velocity rotation
                # Assuming frozen images are strictly 0 and n_images-1 for the flat array
                # product is not frozen
                if i < len(images_mw) - 1 and i not in frozen_indices:
                    chunk_start = movable_idx * (n_atoms * 3)
                    chunk_end = (movable_idx + 1) * (n_atoms * 3)
                    v_shaped = flat_v[chunk_start:chunk_end].reshape(n_atoms, 3)
                    
                    v_rotated = (v_shaped @ _R.T).flatten()
                    flat_v[chunk_start:chunk_end] = v_rotated
                    movable_idx += 1

    if scope == 'global' and opt_state and 'v' in opt_state:
        opt_state['v'] = flat_v
        
    return images_mw









def _neb_interpolate(reactant_cart, product_cart, n_images, masses):
    """
    Linear interpolation between reactant and product in mass-weighted 
    coordinates.
    
    Interpolation in q = sqrt(m) * x space gives a different initial 
    path than Cartesian interpolation: light atoms (H) get proportionally 
    finer resolution because their coordinates are scaled by a smaller 
    sqrt(m). This could a better initial guess for tunneling paths where 
    the dominant motion is hydrogen transfer.
    
    Parameters
    ----------
    reactant_cart : ndarray, shape (N_atoms, 3)
        Reactant coordinates in bohr.
    product_cart : ndarray, shape (N_atoms, 3)
        Product coordinates in bohr.
    n_images : int
        Total number of images INCLUDING the two endpoints.
        
    Returns
    -------
    images_mw : list of ndarray, each (N_atoms, 3)
        Interpolated images in mass-weighted coordinates 
        (bohr * amu^(1/2)).
    """
    sqrt_m = np.sqrt(masses)[:, np.newaxis]  # (N_atoms, 1)
    
    reactant_mw = reactant_cart * sqrt_m
    product_mw  = product_cart  * sqrt_m
    
    images_mw = []
    for i in range(n_images):
        alpha = i / (n_images - 1)
        img = (1.0 - alpha) * reactant_mw + alpha * product_mw
        images_mw.append(img.copy())
    
    return images_mw








def _neb_resample_path(all_coords_bohr, n_new, masses):
    """
    Resample a path to n_new images, equally spaced in mass-weighted 
    arc length.
    
    Uses cubic spline interpolation of the flattened mass-weighted 
    coordinates parameterized by cumulative arc length. Endpoints 
    are preserved exactly.
    
    Parameters
    ----------
    all_coords_bohr : list of ndarray, each (N_atoms, 3)
        Original path in Cartesian (bohr).
    n_new : int
        Target number of images.
        
    Returns
    -------
    resampled : list of ndarray, each (N_atoms, 3)
        Resampled path in Cartesian (bohr).
    """
    n_old = len(all_coords_bohr)
    n_atoms = all_coords_bohr[0].shape[0]
    sqrt_m = np.sqrt(masses)[:, np.newaxis]
    
    # Flatten MW coordinates
    mw_flat = np.array([
        (c * sqrt_m).flatten() for c in all_coords_bohr
    ])
    
    # Cumulative arc length in MW space
    s = np.zeros(n_old)
    for i in range(1, n_old):
        s[i] = s[i-1] + np.linalg.norm(mw_flat[i] - mw_flat[i-1])
    
    if s[-1] < 1e-12:
        print("  WARNING: Path has near-zero length.") # maybe not useful, but could help the user
        return all_coords_bohr[:n_new]
    
    # Normalize to [0, 1]
    s_norm = s / s[-1]
    s_new = np.linspace(0.0, 1.0, n_new)
    
    # Cubic spline of each DOF
    cs = CubicSpline(s_norm, mw_flat, bc_type='clamped')
    mw_new = cs(s_new)
    
    # Force exact endpoints
    mw_new[0]  = mw_flat[0]
    mw_new[-1] = mw_flat[-1]
    
    # Convert back to Cartesian
    resampled = []
    for i in range(n_new):
        mw_img = mw_new[i].reshape(n_atoms, 3)
        resampled.append(mw_img / sqrt_m)
    
    return resampled








def build_initial_band(masses,
                       initial_band=None,
                       reactant_coords=None,
                       product_coords=None,
                       n_images=None,
                       align=True,
                       method='linear',
                       verbose=True):
    """
    Starting band for run_neb, in mass-weighted coordinates.

    Two ways in:

    initial_band
        Any path in Cartesian bohr, shape (N_frames, N_atoms, 3): a rough
        NEB, an IRC run with large steps, a previous MWNEB... If align is
        True, every frame is chain-aligned onto the previous one (reactant
        is the anchor). If n_images is given and differs from N_frames, the
        path is resampled to n_images equally spaced in MW arc length
        (cubic spline, endpoints kept). A 2-frame initial_band is treated
        as reactant + product.

    reactant_coords + product_coords
        The product is aligned to the reactant (mass-weighted Kabsch) and
        the band is built with `method` and n_images (default 15).
        Only 'linear' (interpolation in MW space) for now; other methods
        go here.

    Returns
    -------
    images_mw : list of ndarray (N_atoms, 3), bohr * amu^(1/2)
    """
    masses = np.asarray(masses)
    sqrt_m = np.sqrt(masses)[:, np.newaxis]

    if initial_band is not None:
        frames = [np.asarray(c, dtype=float) for c in initial_band]
        if len(frames) < 2:
            raise ValueError("initial_band needs at least 2 frames.")
        if frames[0].shape != (len(masses), 3):
            raise ValueError(
                f"initial_band frames have shape {frames[0].shape}, expected "
                f"({len(masses)}, 3) for {len(masses)} atoms.")
        if len(frames) == 2:
            reactant_coords, product_coords = frames
        else:
            if align:
                frames = list(geom.align_coordinates(frames, masses)[-1])
                if verbose:
                    print(f"  Aligned {len(frames)} frames (sequential Kabsch).")
            if n_images is not None and n_images != len(frames):
                if verbose:
                    print(f"  Resampled {len(frames)} frames to {n_images} images "
                          f"(equal MW arc length).")
                frames = _neb_resample_path(frames, n_images, masses)
            elif verbose:
                print(f"  Initial band: {len(frames)} images as given.")
            return [c * sqrt_m for c in frames]

    if reactant_coords is None or product_coords is None:
        raise ValueError(
            "Give initial_band, or reactant_coords and product_coords.")

    if n_images is None:
        n_images = 15
    if n_images < 3:
        raise ValueError("n_images must be at least 3 (one movable image).")

    reactant_cart = np.array(reactant_coords)
    product_cart = np.array(product_coords)
    if verbose:
        print("Aligning product to reactant...")
    product_cart, _, _ = geom.kabsch_alignment_cm(product_cart, reactant_cart, masses)

    if method == 'linear':
        images_mw = _neb_interpolate(reactant_cart, product_cart, n_images, masses)
        if verbose:
            print(f"  Cold start: {n_images} images "
                  "(linear interpolation in MW space).")
    else:
        raise ValueError(f"Unknown method '{method}'. Only 'linear' for now.")

    return images_mw









# class BandNebMixin:

    # def _neb_align_band(self, images_mw, frozen_indices=[], opt_state=None, scope='global'):
    #     """Chain-align each movable image onto its left neighbour (mass-weighted,
    #     via Cartesian), in place. Frozen images are anchors. Idempotent.
    #     Applies the rotation matrix to optimizer velocities and L-BFGS history."""
    #     sqrt_m = np.sqrt(self.masses)[:, np.newaxis]
        
    #     # If global scope, the velocities are stored as a single flat array
    #     if scope == 'global' and opt_state and 'v' in opt_state:
    #         n_atoms = len(self.masses)
    #         flat_v = opt_state['v'].copy()
    #         movable_idx = 0 
            
    #     for i in range(1, len(images_mw)):
    #         # SKIP FROZEN IMAGES to prevent anchor drift!
    #         # if i in frozen_indices:
    #         #     continue
                
    #         cart_i   = images_mw[i]     / sqrt_m
    #         cart_ref = images_mw[i - 1] / sqrt_m
            
    #         # ALIGN
    #         cart_aligned, _R, _T = geom.kabsch_alignment_cm(cart_i, cart_ref, self.masses)
    #         images_mw[i] = cart_aligned * sqrt_m
            
    #         # --- APPLY ROTATION TO OPTIMIZER STATE ---
    #         # [TODO] There are problems here (avoided by projecting the tr+rot on ALL
    #         #        forces and not aligning...)
    #         if opt_state:
    #             if scope == 'local' and i in opt_state:
                    
    #                 # A. VPO / FIRE Velocity (Vectors: Rotate only)
    #                 if 'v' in opt_state[i]:
    #                     v_shaped = opt_state[i]['v'].reshape(-1, 3)
    #                     opt_state[i]['v'] = (v_shaped @ _R.T).flatten()
                        
    #                 # B. L-BFGS History
    #                 if 's_hist' in opt_state[i]:
    #                     # History vectors (s, y, prev_g): Rotate only (commutes with mass-weighting)
    #                     opt_state[i]['s_hist'] = [
    #                         (s.reshape(-1, 3) @ _R.T).flatten() for s in opt_state[i]['s_hist']
    #                     ]
    #                     opt_state[i]['y_hist'] = [
    #                         (y.reshape(-1, 3) @ _R.T).flatten() for y in opt_state[i]['y_hist']
    #                     ]
    #                     if 'prev_g' in opt_state[i] and opt_state[i]['prev_g'] is not None:
    #                         opt_state[i]['prev_g'] = (opt_state[i]['prev_g'].reshape(-1, 3) @ _R.T).flatten()
                        
    #                     # Previous coordinate array (prev_q): Convert to Cart -> Rotate & Translate -> Convert to MW
    #                     if 'prev_q' in opt_state[i] and opt_state[i]['prev_q'] is not None:
    #                         cart_prev = opt_state[i]['prev_q'].reshape(-1, 3) / sqrt_m
    #                         cart_prev_aligned = (cart_prev @ _R.T) + _T
    #                         opt_state[i]['prev_q'] = (cart_prev_aligned * sqrt_m).flatten()
                    
    #                 # C. future optimizers here should be added here too [TODO
                    
    #             elif scope == 'global' and 'v' in opt_state:
    #                 # Global scope VPO/FIRE velocity rotation
    #                 # Assuming frozen images are strictly 0 and n_images-1 for the flat array
    #                 # product is not frozen
    #                 if i < len(images_mw) - 1:
    #                     chunk_start = movable_idx * (n_atoms * 3)
    #                     chunk_end = (movable_idx + 1) * (n_atoms * 3)
    #                     v_shaped = flat_v[chunk_start:chunk_end].reshape(n_atoms, 3)
                        
    #                     v_rotated = (v_shaped @ _R.T).flatten()
    #                     flat_v[chunk_start:chunk_end] = v_rotated
    #                     movable_idx += 1

    #     if scope == 'global' and opt_state and 'v' in opt_state:
    #         opt_state['v'] = flat_v
            
    #     return images_mw








    # def _neb_align_and_freeze(self, 
    #                           reactant_cart, 
    #                           product_cart, 
    #                           ts_cart,
    #                           images_mw=None, 
    #                           fmax_tol=4.5e-4,
    #                           validate_ts_gradient=False,
    #                           orca_path='orca', 
    #                           scratch_dir='neb_scratch',
    #                           verbose=True):
    #     """
    #     Align product and TS to reactant (mass-weighted Kabsch), validate the
    #     triple, and identify which image of the band should be frozen at the
    #     TS geometry.

    #     Parameters
    #     ----------
    #     reactant_cart, product_cart, ts_cart : ndarray (N_atoms, 3)
    #         Geometries in bohr. Reactant defines the reference frame.
    #     images_mw : list of ndarray, optional
    #         Existing band in mass-weighted coordinates (warm start). If None,
    #         no snap-to-nearest is performed and ts_idx is returned as None;
    #         the caller is responsible for placing the TS in the cold-start
    #         interpolation.
    #     fmax_tol : float
    #         Convergence tolerance, used as the reference scale for the
    #         opt-in TS gradient sanity check.
    #     validate_ts_gradient : bool
    #         If True, run a single ORCA gradient at the TS geometry and warn
    #         if |grad|_max > 10 * fmax_tol. Adds one ORCA call to startup.
    #     orca_path, scratch_dir : str
    #         Forwarded to the ORCA call when validate_ts_gradient is True.
    #     verbose : bool

    #     Returns
    #     -------
    #     reactant_cart : ndarray (N_atoms, 3)
    #         Unchanged (defines the frame).
    #     product_aligned : ndarray (N_atoms, 3)
    #         Product after mass-weighted Kabsch alignment to reactant.
    #     ts_aligned : ndarray (N_atoms, 3)
    #         TS after mass-weighted Kabsch alignment to reactant.
    #     ts_idx : int or None
    #         If images_mw was provided, the index of the image to freeze
    #         (snap-to-nearest in MW space, lowest index on ties). None
    #         otherwise.
    #     """
    #     # 1. Kabsch alignments
    #     product_aligned, R_p, _ = geom.kabsch_alignment_cm(
    #         product_cart, reactant_cart, self.masses
    #     )
    #     ts_aligned, R_t, _ = geom.kabsch_alignment_cm(
    #         ts_cart, reactant_cart, self.masses
    #     )

    #     # 2. Mass-weighted distances for triangle + snap 
    #     sqrt_m = np.sqrt(self.masses)[:, np.newaxis]
    #     R_mw  = reactant_cart   * sqrt_m
    #     P_mw  = product_aligned * sqrt_m
    #     T_mw  = ts_aligned      * sqrt_m

    #     d_RT = np.linalg.norm((T_mw - R_mw).flatten())
    #     d_TP = np.linalg.norm((P_mw - T_mw).flatten())
    #     d_RP = np.linalg.norm((P_mw - R_mw).flatten())

    #     if verbose:
    #         print(f"  MW distances (bohr*amu^(1/2)):")
    #         print(f"    R -> TS : {d_RT:.4f}")
    #         print(f"    TS -> P : {d_TP:.4f}")
    #         print(f"    R -> P  : {d_RP:.4f}")
    #         print(f"    sum/dRP : {(d_RT + d_TP) / max(d_RP, 1e-12):.4f}")

    #     # Triangle inequality warning. Should be changed in the future
    #     if (d_RT + d_TP) > 1.5 * d_RP:
    #         print(
    #             f"  WARNING: (d(R,TS) + d(TS,P)) / d(R,P) = "
    #             f"{(d_RT + d_TP) / max(d_RP, 1e-12):.3f} > 1.5. "
    #             f"The TS may be rotated relative to the endpoints, the "
    #             f"endpoints may be in inconsistent frames, or the supplied "
    #             f"TS may belong to a different reaction."
    #         )

    #     # Per-atom displacement outlier check warning... qustionable. Probably remove...
    #     disp = np.linalg.norm(P_mw - R_mw, axis=1)  # one number per atom
    #     median_disp = np.median(disp)
    #     if median_disp > 1e-12:
    #         outliers = np.where(disp > 5.0 * median_disp)[0]
    #         if len(outliers) > 0:
    #             symbols = [elem.elements[int(an)][0]
    #                     for an in self.atomic_numbers]
    #             outlier_str = ", ".join(
    #                 f"{symbols[i]}{i} ({disp[i] / median_disp:.1f}x median)"
    #                 for i in outliers
    #             )
    #             print(
    #                 f"  WARNING: atom(s) with MW displacement > 5x median: "
    #                 f"{outlier_str}. If unexpected, check atom ordering "
    #                 f"between reactant and product."
    #             )

    #     # Checking the gradient at the TS... Not used but it's going to be useful in the future...
    #     if validate_ts_gradient:
    #         if verbose:
    #             print("  Validating TS gradient (single ORCA call)...")
    #         ts_check_dir = os.path.join(scratch_dir, 'ts_validate')
    #         os.makedirs(ts_check_dir, exist_ok=True)
    #         # Build a 3-image variable so we can reuse _neb_write_inputs
    #         # (only image index 1 will actually be written)
    #         stub_cart = [reactant_cart, ts_aligned, product_aligned]
    #         files = self._neb_write_inputs(
    #             stub_cart, ts_check_dir, indices=[1]
    #         )
    #         failed = self._neb_run_parallel(orca_path, files, 1)
    #         if failed:
    #             print(
    #                 f"  WARNING: TS gradient validation failed (ORCA error). "
    #                 f"Skipping check. See {ts_check_dir}/"
    #             )
    #         else:
    #             _, forces_mw_stub = self._neb_parse_results(
    #                 ts_check_dir, 3, indices=[1]
    #             )
    #             f_cart_max = np.max(np.abs(
    #                 forces_mw_stub[1] * sqrt_m
    #             ))
    #             if f_cart_max > 10.0 * fmax_tol:
    #                 print(
    #                     f"  WARNING: |grad(TS)|_max = {f_cart_max:.6f} "
    #                     f"Eh/bohr > 10 * fmax_tol "
    #                     f"({10.0 * fmax_tol:.6f}). The supplied TS is not "
    #                     f"a stationary point at this level of theory. "
    #                     f"Consider re-optimizing it with the same ORCA "
    #                     f"settings before running NEB."
    #                 )
    #             elif verbose:
    #                 print(
    #                     f"    |grad(TS)|_max = {f_cart_max:.6f} Eh/bohr  OK"
    #                 )

    #     # Snap the TS to the nearest in MW space 
    #     ts_idx = None
    #     if images_mw is not None:
    #         n = len(images_mw)
    #         dists = np.array([
    #             np.linalg.norm((images_mw[i] - T_mw).flatten())
    #             for i in range(n)
    #         ])
    #         ts_idx = int(np.argmin(dists))  # ties: lowest index (numpy default!)

    #         if ts_idx == 0 or ts_idx == n - 1:
    #             raise RuntimeError(
    #                 f"Nearest image to TS is an endpoint (idx {ts_idx}). "
    #                 f"The warm-start band does not actually cross the TS."
    #             )

    #         if verbose:
    #             print(
    #                 f"  Snapping image {ts_idx} (of {n}) to TS geometry. "
    #                 f"Pre-snap MW distance: {dists[ts_idx]:.4f} "
    #                 f"bohr*amu^(1/2)"
    #             )

    #         # Replace the image's geometry with the (aligned) TS in MW space
    #         images_mw[ts_idx] = T_mw.copy()

    #     return reactant_cart, product_aligned, ts_aligned, ts_idx








    # def _neb_interpolate(self, reactant_cart, product_cart, n_images):
    #     """
    #     Linear interpolation between reactant and product in mass-weighted 
    #     coordinates.
        
    #     Interpolation in q = sqrt(m) * x space gives a different initial 
    #     path than Cartesian interpolation: light atoms (H) get proportionally 
    #     finer resolution because their coordinates are scaled by a smaller 
    #     sqrt(m). This could a better initial guess for tunneling paths where 
    #     the dominant motion is hydrogen transfer.
        
    #     Parameters
    #     ----------
    #     reactant_cart : ndarray, shape (N_atoms, 3)
    #         Reactant coordinates in bohr.
    #     product_cart : ndarray, shape (N_atoms, 3)
    #         Product coordinates in bohr.
    #     n_images : int
    #         Total number of images INCLUDING the two endpoints.
            
    #     Returns
    #     -------
    #     images_mw : list of ndarray, each (N_atoms, 3)
    #         Interpolated images in mass-weighted coordinates 
    #         (bohr * amu^(1/2)).
    #     """
    #     sqrt_m = np.sqrt(self.masses)[:, np.newaxis]  # (N_atoms, 1)
        
    #     reactant_mw = reactant_cart * sqrt_m
    #     product_mw  = product_cart  * sqrt_m
        
    #     images_mw = []
    #     for i in range(n_images):
    #         alpha = i / (n_images - 1)
    #         img = (1.0 - alpha) * reactant_mw + alpha * product_mw
    #         images_mw.append(img.copy())
        
    #     return images_mw








    # def _neb_interpolate_via_ts(self, reactant_cart, product_cart,
    #                             ts_cart, n_images, verbose=True):
    #     """
    #     Two-segment linear interpolation R -> TS -> P in mass-weighted
    #     coordinates. The TS is placed at the image whose fractional position
    #     along the band best matches the MW arc-length fraction d(R,TS) /
    #     [d(R,TS) + d(TS,P)].

    #     Parameters
    #     ----------
    #     reactant_cart, product_cart, ts_cart : ndarray (N_atoms, 3)
    #         Bohr. Should already be aligned (call _neb_align_and_freeze first).
    #     n_images : int
    #         Total number of images including endpoints. Must be >= 5.

    #     Returns
    #     -------
    #     images_mw : list of ndarray
    #         Length n_images. Image 0 = reactant, image n-1 = product, image
    #         ts_idx = TS exactly. The two halves are linearly spaced in MW.
    #     ts_idx : int
    #         Index of the TS image.
    #     """
    #     if n_images < 5:
    #         raise ValueError(
    #             f"n_images = {n_images} too small for via-TS interpolation. "
    #             f"Need >= 5 (R, >=1 left, TS, >=1 right, P)."
    #         )

    #     sqrt_m = np.sqrt(self.masses)[:, np.newaxis]
    #     R_mw = reactant_cart * sqrt_m
    #     P_mw = product_cart  * sqrt_m
    #     T_mw = ts_cart       * sqrt_m

    #     d_RT = np.linalg.norm((T_mw - R_mw).flatten())
    #     d_TP = np.linalg.norm((P_mw - T_mw).flatten())
    #     frac = d_RT / (d_RT + d_TP)

    #     # Choose ts_idx so that ts_idx / (n_images - 1) is closest to frac,
    #     # but ensure at least one movable image on each side.
    #     ts_idx = int(round(frac * (n_images - 1)))
    #     ts_idx = max(2, min(n_images - 3, ts_idx))

    #     images_mw = [None] * n_images
    #     images_mw[0]        = R_mw.copy()
    #     images_mw[ts_idx]   = T_mw.copy()
    #     images_mw[-1]       = P_mw.copy()

    #     # Reactant -> TS
    #     for i in range(1, ts_idx):  
    #         a = i / ts_idx
    #         images_mw[i] = (1.0 - a) * R_mw + a * T_mw

    #     # TS -> Product
    #     for i in range(ts_idx + 1, n_images - 1):
    #         a = (i - ts_idx) / (n_images - 1 - ts_idx)
    #         images_mw[i] = (1.0 - a) * T_mw + a * P_mw

    #     if verbose:
    #         print(
    #             f"  Cold start: {n_images} images, TS placed at index "
    #             f"{ts_idx} (target arc-length fraction {frac:.3f}, "
    #             f"actual {ts_idx / (n_images - 1):.3f})."
    #         )

    #     return images_mw, ts_idx








    # def _neb_resample_path(self, all_coords_bohr, n_new):
    #     """
    #     Resample a path to n_new images, equally spaced in mass-weighted 
    #     arc length.
        
    #     Uses cubic spline interpolation of the flattened mass-weighted 
    #     coordinates parameterized by cumulative arc length. Endpoints 
    #     are preserved exactly.
        
    #     Parameters
    #     ----------
    #     all_coords_bohr : list of ndarray, each (N_atoms, 3)
    #         Original path in Cartesian (bohr).
    #     n_new : int
    #         Target number of images.
            
    #     Returns
    #     -------
    #     resampled : list of ndarray, each (N_atoms, 3)
    #         Resampled path in Cartesian (bohr).
    #     """
    #     n_old = len(all_coords_bohr)
    #     n_atoms = all_coords_bohr[0].shape[0]
    #     sqrt_m = np.sqrt(self.masses)[:, np.newaxis]
        
    #     # Flatten MW coordinates
    #     mw_flat = np.array([
    #         (c * sqrt_m).flatten() for c in all_coords_bohr
    #     ])
        
    #     # Cumulative arc length in MW space
    #     s = np.zeros(n_old)
    #     for i in range(1, n_old):
    #         s[i] = s[i-1] + np.linalg.norm(mw_flat[i] - mw_flat[i-1])
        
    #     if s[-1] < 1e-12:
    #         print("  WARNING: Path has near-zero length.") # maybe not useful, but could help the user
    #         return all_coords_bohr[:n_new]
        
    #     # Normalize to [0, 1]
    #     s_norm = s / s[-1]
    #     s_new = np.linspace(0.0, 1.0, n_new)
        
    #     # Cubic spline of each DOF
    #     cs = CubicSpline(s_norm, mw_flat, bc_type='clamped')
    #     mw_new = cs(s_new)
        
    #     # Force exact endpoints
    #     mw_new[0]  = mw_flat[0]
    #     mw_new[-1] = mw_flat[-1]
        
    #     # Convert back to Cartesian
    #     resampled = []
    #     for i in range(n_new):
    #         mw_img = mw_new[i].reshape(n_atoms, 3)
    #         resampled.append(mw_img / sqrt_m)
        
    #     return resampled