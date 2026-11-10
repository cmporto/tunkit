"""
MW NEB runner, in which run_neb runs the whole optimization
The energy and gradient come from neb_engine.py and ORCA, by
default, an analytic potential, or a new implementation of 
other electronic structure program in the future, as long as
it has the same signature . 

The reactant (image 0) is the alignment anchor. Reactant (image 0)
and product (image N-1) are not moved by the optimizer and their 
energies are computed only once.
"""

import functools
import os

#fuckling imports REMOVE
import functools
import json
import os
import shutil
import tempfile

import numpy as np

from tunkit.core import constants as const
from tunkit.core import geometry as geom
from tunkit.core import engine
from tunkit.path import diagnostics
from tunkit.path.reaction_path import ReactionPath
from tunkit.neb import band
from tunkit.neb import neb_engine
from tunkit.neb import forces
from tunkit.neb import neb_io
from tunkit.neb import optimizers
from tunkit.neb.neb_results import NebResult









def run_neb(path,
            header=None,
            initial_band=None,
            reactant_coords=None,
            product_coords=None,
            n_images=None,
            align_initial_band=True,
            interpolation='linear',
            calculator=None,
            k_spring=0.1,
            max_steps=300,
            fmax_tol=4.5e-4,
            frms_tol=3.0e-4,
            optimizer='vpo',
            scope='global',
            split_forces=False,
            precondition=False,
            max_move=0.2,
            lbfgs_mem=20,
            lbfgs_dr=1.0e-3,
            align_every_step=False,
            orca_path='orca',
            n_parallel=4,
            scratch_dir='neb_scratch',
            job_prefix=None,
            save_every=10,
            verbose=True):
    """
    Run a mass-weighted Nudged Elastic Band optimization.

    Parameters
    ----------
    path : ReactionPath
        Supplies the molecule: path.masses (isotopes included) and
        path.atomic_numbers. Its geometries are NOT used unless passed as
        initial_band (e.g. initial_band=path.raw_coord).
    header : str
        ORCA header from ee_inputs.make_header_orca(make_header_neb=True).
        Needed for the default ORCA calculator.
    initial_band : ndarray (N_frames, N_atoms, 3), optional
        Starting band in bohr: a rough NEB, a large-step IRC, a previous
        run... See band.build_initial_band for alignment and resampling.
    reactant_coords, product_coords : ndarray (N_atoms, 3), optional
        Bohr. Used when initial_band is None: the band is built by
        `interpolation` between them.
    n_images : int, optional
        Images including endpoints. With reactant/product: default 15.
        With initial_band: resample to this many (None keeps its frames).
    align_initial_band : bool
        Chain-align initial_band before use. Default True.
    interpolation : str
        How to build the band from reactant + product. 'linear' (in MW space).
    calculator : callable, optional
        calculator(images_cart, indices, work_dir) -> (energies, gradients).
        Default: ORCA via neb_engine.run_orca_engrad with header, orca_path, n_parallel.
    k_spring : float
        Spring constant in Eh / (bohr * amu^(1/2))^2.
    max_steps : int
    fmax_tol, frms_tol : float
        Convergence on perpendicular Cartesian force (Eh/bohr).
    optimizer : str
        'vpo'   - Velocity Projected Optimizer (default).
        'fire'  - FIRE (Bitzek 2006).
        'lbfgs' - L-BFGS with trust radius (always per image).
    scope : str
        'global' - whole band is one optimizer vector (default).
        'local'  - per-image optimizer state. Ignored for 'lbfgs'.
    split_forces : bool
        VPO/FIRE build momentum only on F_perp; F_spring is applied as a
        steepest-descent correction. Default False (combined force).
    precondition : bool
        With split_forces, multiply F_perp by the atom mass before the
        optimizer. Default False.
    max_move : float
        Trust-radius cap on the max step component. Default 0.2.
    lbfgs_mem, lbfgs_dr : L-BFGS history depth and FD probe size.
    align_every_step : bool
        Chain-align the band before every force evaluation (rotating the
        optimizer state with it). Default False: only the product is
        re-aligned onto image N-2 each step.
    orca_path, n_parallel : ORCA executable and concurrent jobs (default calculator).
    scratch_dir : str
        'frozen/' and 'current/' calculator folders, restart.json, band.xyz.
    job_prefix : str, optional
        If set, band files are <job_prefix>_band.xyz / _all_bands.xyz.
    save_every : int
        Save restart + band every N steps (0 = only at the end).
    verbose : bool

    Returns
    -------
    NebResult
        result.path is a new ReactionPath with the final band (raw_coord,
        raw_coord_ang, mass_scaled_coord, energies) and the input path's
        masses and atomic numbers.
    """

    # =============================================================
    #  0. Validate inputs
    # =============================================================
    if optimizer not in ('vpo', 'fire', 'lbfgs'):
        raise ValueError(
            f"Unknown optimizer '{optimizer}'. Use 'vpo', 'fire', or 'lbfgs'.")
    if scope not in ('global', 'local'):
        raise ValueError(f"Unknown scope '{scope}'. Use 'global' or 'local'.")
    if optimizer == 'lbfgs' and scope == 'global':
        if verbose:
            print("  Note: L-BFGS is always per-image (local). "
                  "Ignoring scope='global'.")
        scope = 'local'
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1.")
    if n_parallel < 1:
        raise ValueError("n_parallel must be >= 1.")
    if path.masses is None or path.atomic_numbers is None:
        raise ValueError(
            "path.masses and path.atomic_numbers are needed. Load the "
            "molecule first (e.g. load_neb_path or extract_coord_energy_xyz).")

    masses = np.asarray(path.masses)
    atomic_numbers = path.atomic_numbers
    sqrt_m = np.sqrt(masses)[:, np.newaxis]

    if calculator is None:
        if not header:
            raise ValueError(
                "header is needed for the ORCA calculator. Use "
                "ee_inputs.make_header_orca(..., make_header_neb=True), "
                "or pass calculator=.")
        calculator = functools.partial(neb_engine.run_orca_engrad,
                                       header=header,
                                       atomic_numbers=atomic_numbers,
                                       orca_path=orca_path,
                                       n_parallel=n_parallel)

    # =============================================================
    #  1. Initial band
    # =============================================================
    images_mw = band.build_initial_band(masses,
                                        initial_band=initial_band,
                                        reactant_coords=reactant_coords,
                                        product_coords=product_coords,
                                        n_images=n_images,
                                        align=align_initial_band,
                                        method=interpolation,
                                        verbose=verbose)
    n_images = len(images_mw)
    frozen_indices = {0, n_images - 1}
    movable = list(range(1, n_images - 1))
    n_movable = len(movable)

    # =============================================================
    #  2. Setup
    # =============================================================
    os.makedirs(scratch_dir, exist_ok=True)

    if scope == 'global' and optimizer in ('vpo', 'fire'):
        opt_state = {}                       # single global state
    else:
        opt_state = {i: {} for i in movable}  # per-image state

    # Preconditioning vector, reused each step
    m_precond = np.tile(np.repeat(masses, 3), n_movable) if precondition else None

    if verbose:
        opt_label = f"{optimizer.upper()} ({scope})"
        if split_forces:
            opt_label += " split"
        if precondition:
            opt_label += "+precond"
        print(f"\n  Optimizer: {opt_label}")
        print(f"  max_move: {max_move:.4f}")

    # =============================================================
    #  3. Endpoint energies (once)
    # =============================================================
    if verbose:
        print("Computing energies of the endpoints...")
    frozen_list = sorted(frozen_indices)
    fr_energies, fr_forces = evaluate_band(images_mw, masses, calculator, frozen_list,
                                           os.path.join(scratch_dir, 'frozen'))
    frozen_E = {i: fr_energies[i] for i in frozen_list}
    frozen_F = {i: fr_forces[i] for i in frozen_list}
    if verbose:
        print(f"  {'Reactant':.<30s}: {frozen_E[0]:.8f} Eh")
        print(f"  {'Product':.<30s}: {frozen_E[n_images - 1]:.8f} Eh")
        check_band_spacing(images_mw, max_move)

    # =============================================================
    #  4. Optimization loop
    # =============================================================
    if verbose:
        _print_table_header(n_images, n_movable, k_spring, fmax_tol, frms_tol)

    neb_converged = False
    fmax_history, frms_history = [], []
    iter_dir = os.path.join(scratch_dir, 'current')

    for step in range(max_steps):
        # 4a. Alignment before the force evaluation
        if align_every_step:
            images_mw = band._neb_align_band(images_mw, masses,
                                             opt_state=opt_state, scope=scope)
        else:
            images_mw = _align_product(images_mw, masses)

        # 4b. Energies and forces of the movable images, endpoints spliced in
        energies, forces_mw = evaluate_band(images_mw, masses, calculator, movable, iter_dir)
        for i, E in frozen_E.items():
            energies[i] = E
        for i, F in frozen_F.items():
            forces_mw[i] = F

        # 4c. NEB forces
        neb_forces_perp, neb_forces_spring, fmax, frms, theta_max, df_err, fmax_per_image = \
            forces._neb_compute_forces(images_mw, energies, forces_mw, k_spring, masses,
                                       frozen_indices=frozen_indices)
        fmax_history.append(fmax)
        frms_history.append(frms)
        if verbose:
            _print_table_row(step, energies, fmax, frms, neb_forces_spring,
                             movable, fmax_per_image, theta_max)

        # 4d. Convergence
        if fmax < fmax_tol and frms < frms_tol:
            neb_converged = True
            if verbose:
                print(f"\n  *** CONVERGED at step {step} ***")
                print(f"  max|F_perp| = {fmax:.6f}  rms|F_perp| = {frms:.6f}")
            break

        # 4e. Optimizer step
        images_mw = optimizer_step(images_mw, neb_forces_perp, neb_forces_spring,
                                   movable, opt_state,
                                   optimizer=optimizer,
                                   scope=scope,
                                   split_forces=split_forces,
                                   m_precond=m_precond,
                                   max_move=max_move,
                                   lbfgs_mem=lbfgs_mem,
                                   lbfgs_dr=lbfgs_dr)

        # 4f. Save periodically
        if save_every > 0 and step % save_every == 0:
            neb_io._neb_save_state(scratch_dir=scratch_dir,
                                   images_mw=images_mw,
                                   energies=energies,
                                   fmax_history=fmax_history,
                                   step=step,
                                   masses=masses,
                                   atomic_numbers=atomic_numbers,
                                   fmax_per_image=fmax_per_image,
                                   job_prefix=job_prefix)
    else:
        if verbose:
            print(f"\n  WARNING: Did not converge in {max_steps} "
                  f"steps (fmax = {fmax:.6f}, frms = {frms:.6f})")

    # =============================================================
    #  5. Results
    # =============================================================
    neb_io._neb_save_state(scratch_dir=scratch_dir,
                           images_mw=images_mw,
                           energies=energies,
                           fmax_history=fmax_history,
                           step=step,
                           masses=masses,
                           atomic_numbers=atomic_numbers,
                           fmax_per_image=fmax_per_image,
                           job_prefix=job_prefix)

    final_cart = np.array([mw / sqrt_m for mw in images_mw])
    new_path = ReactionPath(masses=path.masses,
                            atomic_numbers=path.atomic_numbers,
                            reactant_file=path.reactant_file,
                            raw_coord=final_cart,
                            raw_coord_ang=final_cart * const.bohr_to_ang,
                            mass_scaled_coord=np.array(images_mw),
                            energies=energies)

    result = NebResult(path=new_path,
                       converged=neb_converged,
                       n_steps=len(fmax_history),
                       fmax_history=fmax_history,
                       frms_history=frms_history,
                       fmax_per_image=fmax_per_image,
                       scratch_dir=scratch_dir,
                       params=dict(n_images=n_images, k_spring=k_spring,
                                   max_steps=max_steps, fmax_tol=fmax_tol,
                                   frms_tol=frms_tol, optimizer=optimizer,
                                   scope=scope, split_forces=split_forces,
                                   precondition=precondition, max_move=max_move,
                                   lbfgs_mem=lbfgs_mem, lbfgs_dr=lbfgs_dr,
                                   align_every_step=align_every_step,
                                   align_initial_band=align_initial_band,
                                   interpolation=interpolation))
    if verbose:
        print(f"Final barrier: {result.barrier_kcalmol:.4f} kcal/mol "
              f"(highest image: {result.highest_image})")
        print("Results in result.path (raw_coord, mass_scaled_coord, energies).")
    return result








def evaluate_band(images_mw, masses, engine, indices, work_dir):
    """
    Energies and mass-weighted forces of the band images.

    Parameters
    ----------
    images_mw : list of ndarray (N_atoms, 3)
        The whole band, mass-weighted.
    engine : callable
        engine(images_cart, indices, work_dir) -> (energies, gradients).
    indices : list of int
        Images to compute.

    Returns
    -------
    energies : ndarray (N_images,)
        Eh. Only entries in indices are filled; the rest are zero.
    forces_mw : list of ndarray (N_atoms, 3)
        F_mw = -gradient / sqrt(m). Only entries in indices are filled.
    """
    masses = np.asarray(masses)
    n_images = len(images_mw)
    n_atoms = len(masses)
    sqrt_m = np.sqrt(masses)[:, np.newaxis]

    images_cart = [mw / sqrt_m for mw in images_mw]
    e_k, g_k = engine(images_cart, list(indices), work_dir)

    energies = np.zeros(n_images)
    forces_mw = [np.zeros((n_atoms, 3)) for _ in range(n_images)]
    for k, i in enumerate(indices):
        energies[i] = e_k[k]
        f_cart = -np.asarray(g_k[k]).reshape(n_atoms, 3)
        forces_mw[i] = f_cart / sqrt_m
    return energies, forces_mw








def optimizer_step(images_mw,
                   neb_forces_perp,
                   neb_forces_spring,
                   movable,
                   opt_state,
                   optimizer='vpo',
                   scope='global',
                   split_forces=False,
                   m_precond=None,
                   max_move=0.2,
                   lbfgs_mem=20,
                   lbfgs_dr=1.0e-3):
    """
    One optimizer step on the movable images.

    opt_state is mutated in place: a single dict for global VPO/FIRE, or
    {image: dict} for local VPO/FIRE and L-BFGS. Returns a new list of
    images (the input list is not modified).
    """
    n_atoms = images_mw[0].shape[0]
    chunk = n_atoms * 3
    positions = np.concatenate([images_mw[i].flatten() for i in movable])

    if split_forces and optimizer in ('vpo', 'fire'):
        # SPLIT MODE: F_perp goes to the optimizer, F_spring is steepest descent
        forces_perp_flat = np.concatenate(
            [neb_forces_perp[i].flatten() for i in movable])
        if m_precond is not None:
            forces_perp_flat = forces_perp_flat * m_precond

        if scope == 'global':
            if optimizer == 'vpo':
                new_pos = optimizers._neb_vpo_step(positions, forces_perp_flat, opt_state,
                                                   dt=0.5, max_move=max_move)
            else:
                new_pos = optimizers._neb_fire_step(positions, forces_perp_flat, opt_state,
                                                    max_move=max_move)
            step_opt = new_pos - positions
        else:
            step_opt = np.zeros_like(positions)
            for j, i in enumerate(movable):
                p_i = positions[j * chunk:(j + 1) * chunk]
                f_i = forces_perp_flat[j * chunk:(j + 1) * chunk]
                if optimizer == 'vpo':
                    new_p_i = optimizers._neb_vpo_step(p_i, f_i, opt_state[i],
                                                       dt=0.5, max_move=max_move)
                else:
                    new_p_i = optimizers._neb_fire_step(p_i, f_i, opt_state[i],
                                                        max_move=max_move)
                step_opt[j * chunk:(j + 1) * chunk] = new_p_i - p_i

        # Spring: gentle steepest descent
        forces_spring_flat = np.concatenate(
            [neb_forces_spring[i].flatten() for i in movable])
        spring_step_size = 0.25
        step_spring = forces_spring_flat * spring_step_size
        spring_max = float(np.max(np.abs(step_spring)))
        if spring_max > 0.5 * max_move:
            step_spring *= 0.5 * max_move / spring_max

        new_positions = positions + step_opt + step_spring

    else:
        # COMBINED MODE: the optimizer handles perp + spring
        neb_forces_combined = {i: neb_forces_perp[i] + neb_forces_spring[i]
                               for i in movable}

        if optimizer == 'lbfgs':
            new_positions = np.empty_like(positions)
            for j, i in enumerate(movable):
                pos_i = images_mw[i].flatten()
                f_i = neb_forces_combined[i].flatten()
                new_positions[j * chunk:(j + 1) * chunk] = \
                    optimizers._neb_lbfgs_step_per_image(pos_i, f_i, opt_state[i],
                                                         mem=lbfgs_mem, dr=lbfgs_dr,
                                                         max_move=max_move)
        else:
            forces_flat = np.concatenate(
                [neb_forces_combined[i].flatten() for i in movable])
            if scope == 'global':
                if optimizer == 'vpo':
                    new_positions = optimizers._neb_vpo_step(positions, forces_flat, opt_state,
                                                             dt=0.5, max_move=max_move)
                else:
                    new_positions = optimizers._neb_fire_step(positions, forces_flat, opt_state,
                                                              max_move=max_move)
            else:
                new_positions = np.empty_like(positions)
                for j, i in enumerate(movable):
                    p_i = positions[j * chunk:(j + 1) * chunk]
                    f_i = forces_flat[j * chunk:(j + 1) * chunk]
                    if optimizer == 'vpo':
                        new_positions[j * chunk:(j + 1) * chunk] = \
                            optimizers._neb_vpo_step(p_i, f_i, opt_state[i],
                                                     dt=0.5, max_move=max_move)
                    else:
                        new_positions[j * chunk:(j + 1) * chunk] = \
                            optimizers._neb_fire_step(p_i, f_i, opt_state[i],
                                                      max_move=max_move)

    # Unflatten back into images
    new_images = list(images_mw)
    for j, i in enumerate(movable):
        new_images[i] = new_positions[j * chunk:(j + 1) * chunk].reshape(n_atoms, 3)
    return new_images








def check_band_spacing(images_mw, max_move):
    """Warn if the trust radius is large compared with the image spacing."""
    print("\nChecking initial path density against trust radius...")
    arc_lengths = diagnostics._calc_arc_lengths(images_mw)
    min_ds = min(arc_lengths)
    avg_ds = sum(arc_lengths) / len(arc_lengths)
    print(f"  Min arc length: {min_ds:.4f} | Avg arc length: {avg_ds:.4f} "
          f"| max_move: {max_move:.4f}")
    if max_move > min_ds / 2.0:
        print(f"[WARNING] Trust radius (max_move={max_move:.4f}) is LARGER \n"
              f"than half the minimum arc length ({min_ds:.4f}).")
        print("Images can jump over each other in a single step.")
        print("This may cause knots in the band or stall the optimization.")
        print("Consider DECREASING max_move or using FEWER images.")








def _align_product(images_mw, masses):
    """Re-align the product (fixed, not an anchor) onto image N-2. Returns a new list."""
    sqrt_m = np.sqrt(masses)[:, np.newaxis]
    cart_n_minus_2 = images_mw[-2] / sqrt_m
    cart_product = images_mw[-1] / sqrt_m
    cart_prod_aligned, _, _ = geom.kabsch_alignment_cm(cart_product, cart_n_minus_2, masses)
    new_images = list(images_mw)
    new_images[-1] = cart_prod_aligned * sqrt_m
    return new_images








def _print_table_header(n_images, n_movable, k_spring, fmax_tol, frms_tol):
    print(f"\n{'=' * 85}")
    print(" Mass-Weighted NEB")
    print(f" Images: {n_images}  Movable: {n_movable}  k = {k_spring}")
    print(f" Convergence: max|F_perp| < {fmax_tol:.1e}, "
          f"rms|F_perp| < {frms_tol:.1e} (Eh/bohr)")
    print(f"{'=' * 85}")
    print(f"{'Step':>5s} {'E_TS':>14s} {'fmax':>12s} {'frms':>12s} "
          f"{'fpar':>12s} {'MaxImg':>8s} {'θ_max(°)':>10s}")
    print('-' * 85)








def _print_table_row(step, energies, fmax, frms, neb_forces_spring, movable,
                     fmax_per_image, theta_max):
    forces_spring_flat = np.concatenate(
        [neb_forces_spring[i].flatten() for i in movable if i in neb_forces_spring])
    fpar = float(np.max(np.abs(forces_spring_flat))) if len(forces_spring_flat) > 0 else 0.0
    max_img_idx = max(fmax_per_image, key=fmax_per_image.get) if fmax_per_image else 0
    print(f"{step:5d} {np.max(energies):14.8f} {fmax:12.6f} {frms:12.6f} "
          f"{fpar:12.6f} {max_img_idx:8d} {theta_max:10.4f}")









def run_neb_from_json(json_file, tol=1e-7, verbose=False, **overrides):
    """
    Re-run a recorded MWNEB from a JSON file, without ORCA.

    The file (written by the developer scripts that save a run) holds the
    molecule ('masses', 'atomic_numbers'), the run_neb options of the
    original run ('run_neb': starting band or endpoints, optimizer settings,
    max_steps) and every calculation it made ('calculations', read by
    neb_engine.run_json_engrad). The run starts from the same band and gets its
    energies and gradients from the file; if at any step it asks for a
    geometry more than tol (bohr) from every recorded one, it stops with an
    error naming the image.

    overrides replace run_neb options, e.g. to see where a different
    k_spring leaves the recorded path.

    Returns
    -------
    NebResult (its scratch folder is removed afterwards)
    """
    with open(json_file) as fo:
        rec = json.load(fo)
    options = dict(rec['run_neb'])
    for key in ('initial_band', 'reactant_coords', 'product_coords'):
        if options.get(key) is not None:
            options[key] = np.array(options[key])
    options.update(overrides)
    path = ReactionPath(masses=np.array(rec['masses']),
                        atomic_numbers=np.array(rec['atomic_numbers']))
    calculator = functools.partial(engine.run_json_engrad,
                                   json_file=os.path.abspath(json_file), tol=tol)
    scratch = tempfile.mkdtemp(prefix='neb_from_json_')
    try:
        return run_neb(path, calculator=calculator, scratch_dir=scratch,
                       verbose=verbose, **options)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

















class RunNebMixin:

    def run_neb(self,
                reactant_coords=None,
                product_coords=None,
                ts_coords=None,
                ts_mode=None,
                n_images=15,
                k_spring=0.1,
                max_steps=300,
                fmax_tol=4.5e-4,
                frms_tol=3.0e-4,
                optimizer='vpo',
                orca_vpo=False,
                orca_vpo_stepsize=0.5,
                scope='global',
                split_forces=False,
                precondition=False,
                max_move=0.2,
                lbfgs_mem=20,
                lbfgs_dr=1.0e-3,
                validate_ts_gradient=False,
                orca_path='orca',
                n_parallel=4,
                scratch_dir='neb_scratch',
                job_prefix=None,
                save_every=10,
                skip_alignment=False,
                align_every_step=False,
                verbose=True):
        """
        Run a mass-weighted Nudged Elastic Band optimization.
    
        Parameters
        ----------
        reactant_coords, product_coords : ndarray (N_atoms, 3), optional
            Geometries in bohr. Required for cold start.
        ts_coords : ndarray (N_atoms, 3), optional
            TS geometry in bohr. Required if ts_mode is not None.
        ts_mode : str or None
            'frozen'  - TS snapped into band, position held fixed.
            'free'    - TS snapped into band as initial guess, free to move.
            None      - no TS supplied, plain NEB (no climbing image). (default)
        n_images : int
            Total images including endpoints. Cold start only.
        k_spring : float
            Spring constant in Eh / (bohr * amu^(1/2))^2.
        max_steps : int
        fmax_tol, frms_tol : float
            Convergence on perpendicular Cartesian force (Eh/bohr).
        optimizer : str
            'vpo'   - Velocity Projected Optimizer (default, recommended).
            'fire'  - FIRE (Bitzek 2006).
            'lbfgs' - L-BFGS with trust radius (experimental).
        scope : str
            'global' - whole band is one optimizer vector (default).
            'local'  - per-image independent optimizer state.
            Ignored for 'lbfgs' (always local).
        split_forces : bool
            If True, perpendicular and spring forces are treated separately:
            the optimizer (VPO/FIRE) builds momentum only on F_perp, and
            F_spring is applied as a gentle steepest-descent correction.
            Default False (combined force).
        precondition : bool
            If True, multiply F_perp by atom mass before passing to the
            optimizer, compensating for the 1/m suppression of heavy-atom
            Cartesian displacements in MW space. Only used when
            split_forces=True and optimizer is 'vpo' or 'fire'.
            Default False.
        max_move : float
            Trust-radius cap on max step component. Default 0.2.
        lbfgs_mem, lbfgs_dr : float
            L-BFGS-specific parameters.
        validate_ts_gradient : bool
            Opt-in: run one ORCA gradient at the TS.
        skip_alignment : bool
            Skip Kabsch alignment on warm-start reload.
        orca_path, n_parallel, scratch_dir, save_every, verbose : ...
        """

    
        # =============================================================
        #  Validate inputs (should add the ones missing)
        # =============================================================
        if ts_mode not in (None, 'frozen', 'free'): 
            raise ValueError("ts_mode must be None, 'frozen', or 'free'.")
        if ts_mode in ('frozen', 'free') and ts_coords is None:
            raise ValueError(
                f"ts_mode='{ts_mode}' requires ts_coords. Supply a TS "
                "geometry, or set ts_mode=None for plain NEB."
            )
        if optimizer not in ('vpo', 'fire', 'lbfgs'):
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. Use 'vpo', 'fire', or 'lbfgs'."
            )
        if scope not in ('global', 'local'):
            raise ValueError(
                f"Unknown scope '{scope}'. Use 'global' or 'local'."
            )
        if optimizer == 'lbfgs' and scope == 'global':
            if verbose:
                print("  Note: L-BFGS is always per-image (local). "
                    "Ignoring scope='global'.")
            scope = 'local'
    
        ts_cart = np.array(ts_coords) if ts_coords is not None else None

        if n_parallel < 1:
            raise ValueError("n_parallel must be >= 1.")
    
        # =============================================================
        #  1. Determine the initial band
        # =============================================================
        
        # [TODO] warm_start checks if there are mass weighted neb images, so there
        # could be an issue of a band being initialized and not register because
        # there were no images
        warm_start = (
            hasattr(self, 'neb_images_mw') and self.neb_images_mw is not None
        )
        sqrt_m = np.sqrt(self.masses)[:, np.newaxis]
    
        if warm_start:
            images_mw = [img.copy() for img in self.neb_images_mw]
            n_images = len(images_mw)
            reactant_cart = images_mw[0] / sqrt_m
            product_cart  = images_mw[-1] / sqrt_m
            if verbose:
                print(f"Warm start: using {n_images} loaded images.")
        else:
            if reactant_coords is None or product_coords is None:
                raise ValueError(
                    "Cold start requires reactant_coords and product_coords."
                )
            reactant_cart = np.array(reactant_coords)
            product_cart  = np.array(product_coords)
    
        # =============================================================
        #  2. Alignment + TS placement ([TODO])
        # =============================================================

        # [TODO] There are no options for ts_mode None and warm_start...
        # I'd like to change the amount of images, and it has been delegated
        # to the generate_neb method. It would be much easier if it were here 
        # as the user could run each part on the CLI
        ts_idx = None
    
        if ts_mode is not None:
            # We have a TS to place
            if skip_alignment and warm_start:
                if verbose:
                    print("Skipping alignment (warm restart).")
                # Just find nearest image to snap
                T_mw = ts_cart * sqrt_m
                dists = np.array([
                    np.linalg.norm((images_mw[i] - T_mw).flatten())
                    for i in range(n_images)
                ])
                ts_idx = int(np.argmin(dists))
                if ts_idx == 0 or ts_idx == n_images - 1:
                    raise RuntimeError(
                        f"Nearest image to TS is an endpoint (idx {ts_idx})."
                    )
                ts_aligned = ts_cart
                if ts_mode == 'frozen':
                    images_mw[ts_idx] = T_mw.copy()
                if verbose:
                    print(f"  TS nearest image: {ts_idx}, "
                        f"dist = {dists[ts_idx]:.6f}")
            else:
                if verbose:
                    print("Aligning product and TS to reactant "
                        "(mass-weighted Kabsch)...")
    
                if warm_start:
                    # If we leave the alignment of product to reactant, it moves the (frozen, anchored) 
                    # product into a different frame than the band converged in, and since aligning the band
                    # alignment won't move an anchor, the tangent at image n-2 will still jump. 
                    # product_cart, _, _ = geom.kabsch_alignment_cm(
                    #     product_cart, reactant_cart, self.masses
                    # )
                    images_mw[-1] = product_cart * sqrt_m
    
                    _, _, ts_aligned, ts_idx = optimizers._neb_align_and_freeze(
                        reactant_cart, product_cart, ts_cart,
                        self.masses, self.atomic_numbers,
                        images_mw=images_mw,
                        fmax_tol=fmax_tol,
                        validate_ts_gradient=validate_ts_gradient,
                        orca_path=orca_path,
                        scratch_dir=scratch_dir,
                        verbose=verbose,
                    )
                    # If free mode, don't snap 
                    if ts_mode == 'free':
                        # Keep the snapped geometry as is - it's a good
                        # starting point... the image will be free to move.
                        pass
                else:
                    _, product_aligned, ts_aligned, _ = \
                        optimizers._neb_align_and_freeze(
                            reactant_cart, product_cart, ts_cart,
                            self.masses, self.atomic_numbers,
                            images_mw=None,
                            fmax_tol=fmax_tol,
                            validate_ts_gradient=validate_ts_gradient,
                            orca_path=orca_path,
                            scratch_dir=scratch_dir,
                            verbose=verbose,
                        )
                    product_cart = product_aligned
                    images_mw, ts_idx = optimizers._neb_interpolate_via_ts(
                        reactant_cart, 
                        product_cart, 
                        ts_aligned, 
                        n_images,
                        self.masses,
                        verbose=verbose,
                    )
        else:
            # No TS - plain NEB
            if not warm_start:
                if verbose:
                    print("Aligning product to reactant...")
                product_cart, _, _ = geom.kabsch_alignment_cm(
                    product_cart, reactant_cart, self.masses
                )
                images_mw = optimizers._neb_interpolate(
                    reactant_cart, product_cart, n_images, self.masses
                )
                if verbose:
                    print(f"  Cold start: {n_images} images "
                        "(linear interpolation in MW space).")
    
        # Build frozen set
        if ts_mode == 'frozen' and ts_idx is not None:
            frozen_indices = {0, ts_idx, n_images - 1}
        else:
            frozen_indices = {0, n_images - 1}
    
        if verbose:
            print(f"Frozen image indices: {sorted(frozen_indices)}")
            if ts_mode == 'free' and ts_idx is not None:
                print(f"  (TS at idx {ts_idx}, free to move)")
            if ts_mode is None:
                print(f"  (No TS supplied, plain NEB)")
    
        # =============================================================
        #  3. Setup
        # =============================================================
        n_atoms = len(self.masses)
        effective_max_move = float(max_move)
    
        if not hasattr(self, 'neb_header') or not self.neb_header:
            raise ValueError(
                "self.neb_header is not set. Call make_header_orca() "
                "with the NEB flag before run_neb()."
            )
    
        os.makedirs(scratch_dir, exist_ok=True)
    
        # Optimizer state
        movable = [i for i in range(1, n_images - 1)
                if i not in frozen_indices]
        n_movable = len(movable)
    
        if scope == 'global' and optimizer in ('vpo', 'fire'):
            opt_state = {}  # single global state
        else:
            # Per-image state (local VPO, local FIRE, or L-BFGS)
            opt_state = {i: {} for i in movable}
    
        # Preconditioning vector, reused each step (perhaps could be changed as the optimization continues...)
        if precondition:
            m_precond = np.tile(
                np.repeat(self.masses, 3), n_movable
            )
        else:
            m_precond = None
    
        neb_converged = False
        fmax_history = []
    
        if verbose:
            opt_label = f"{optimizer.upper()} ({scope})"
            if split_forces:
                opt_label += " split"
            if precondition:
                opt_label += "+precond"
            print(f"\n  Optimizer: {opt_label}")
            print(f"  max_move: {effective_max_move:.4f}")
    
        # =============================================================
        #  4. Compute energies of frozen images (once)
        # =============================================================
        if verbose:
            print(f"Computing energies of frozen images...")
    
        frozen_list = sorted(frozen_indices)
        images_cart = [mw / sqrt_m for mw in images_mw]
        frozen_dir = os.path.join(scratch_dir, 'frozen')
        frozen_files = self._neb_write_inputs(
            images_cart, frozen_dir, indices=frozen_list
        )
        failed = engine.run_orca_parallel(
            orca_path, frozen_files, min(n_parallel, len(frozen_list))
        )
        if failed:
            raise RuntimeError(
                f"ORCA failed for frozen image(s) {failed}. "
                f"Check {frozen_dir}/"
            )
    
        fr_energies, fr_forces = neb_io._neb_parse_results(
            frozen_dir, n_images, indices=frozen_list
        )
        frozen_E = {i: fr_energies[i] for i in frozen_list}
        frozen_F = {i: fr_forces[i]   for i in frozen_list}
    
        if verbose:
            for fi in frozen_list:
                label = "Reactant" if fi == 0 else (
                    "Product" if fi == n_images - 1 else f"TS (idx {fi})"
                )
                print(f"  {label:.<30s}: {frozen_E[fi]:.8f} Eh")
            if ts_idx is not None and ts_idx in frozen_E:
                bh = (frozen_E[ts_idx] - frozen_E[0]) * const.hartree_to_kcalmol
                print(f"  Fixed barrier height: {bh:.4f} kcal/mol")
                if frozen_E[ts_idx] < max(
                    frozen_E[0], frozen_E[n_images - 1]
                ) + 0.5 / const.hartree_to_kcalmol:
                    print(
                        "  WARNING: TS energy is not at least 0.5 kcal/mol "
                        "above both endpoints."
                    )

        # =============================================================
        #  4.5 Pre-Optimization Health Check (added later)
        # =============================================================

        if verbose:
            print("\nChecking initial path density against trust radius...")

            arc_lengths = diagnostics._calc_arc_lengths(images_mw,)
            # arc_lengths = []
            # for i in range(n_images - 1):
            #     ds = np.linalg.norm((images_mw[i+1] - images_mw[i]).flatten())
            #     arc_lengths.append(ds)
            
            min_ds = min(arc_lengths)
            avg_ds = sum(arc_lengths) / len(arc_lengths)
            
            print(f"  Min arc length: {min_ds:.4f} | Avg arc length: {avg_ds:.4f} | max_move: {effective_max_move:.4f}")
            
            # The trust radius should be smaller than the distance between images
            if effective_max_move > min_ds / 2.0:
                print(f"[WARNING] Trust radius (max_move={effective_max_move:.4f}) is LARGER \nthan the minimum arc length ({min_ds:.4f}).")
                print("Images can jump over each other in a single step.")
                print("This may cause knots in the band or stall the optimization.")
                print("Consider DECREASING max_move or using FEWER images.") # it should be changed as we change the method...

        # =============================================================
        #  5. Optimization loop
        # =============================================================
        if verbose:
            print(f"\n{'='*85}")
            print(f" Mass-Weighted NEB")
            print(f" Images: {n_images}  Movable: {n_movable}  "
                f"k = {k_spring}")
            print(f" Convergence: max|F_perp| < {fmax_tol:.1e}, "
                f"rms|F_perp| < {frms_tol:.1e} (Eh/bohr)")
            print(f"{'='*85}")
            header = f"{'Step':>5s} {'E_TS':>14s} {'fmax':>12s} {'frms':>12s} {'fpar':>12s} {'MaxImg':>8s} {'θ_max(°)':>10s}" # {'ΔF_err':>10s}"
            print(header)
            print('-' * 85)
    
        for step in range(max_steps):
            # Align band before the force evaluation (there are issues with the optimizer's history)
            if align_every_step:
                images_mw = self._neb_align_band(images_mw, self.masses, opt_state=opt_state, scope=scope)
            # The other possibility being tested is projecting out
            # the rotations and translations of all forces before applying them (seems to work well on most cases)
            else:
                # Because of the froze TS algorithm before, this part is necessary
                # Check if the there isn't a frozen TS or if the product isn't it
                if ts_mode != 'frozen' or ts_idx != n_images - 1:
                    cart_n_minus_2 = images_mw[-2] / sqrt_m
                    cart_product   = images_mw[-1] / sqrt_m
                    
                    # Align Product to N-2
                    cart_prod_aligned, _, _ = geom.kabsch_alignment_cm(cart_product, cart_n_minus_2, self.masses)
                    images_mw[-1] = cart_prod_aligned * sqrt_m

            # 5a. Write + run + parse 
            images_cart = [mw / sqrt_m for mw in images_mw]
            iter_dir = os.path.join(scratch_dir, 'current')
    
            # For ts_mode='free', also compute TS image each step
            compute_indices = list(movable)
            if ts_mode == 'free' and ts_idx is not None and ts_idx not in compute_indices:
                compute_indices.append(ts_idx)
                compute_indices.sort()
    
            inp_files = self._neb_write_inputs(
                images_cart, iter_dir, indices=compute_indices
            )
            failed = engine.run_orca_parallel(
                orca_path, inp_files, n_parallel
            )
            if failed:
                raise RuntimeError(f"  Step {step}: ORCA failed for images {failed}. "
                    f"Stopping.")


            energies, forces_mw = neb_io._neb_parse_results(
                iter_dir, n_images, indices=compute_indices
            )
    
            # Splice frozen data
            for i, E in frozen_E.items():
                energies[i] = E
            for i, F in frozen_F.items():
                forces_mw[i] = F
    
            # 5b. NEB forces 
            neb_forces_perp, neb_forces_spring, fmax, frms, theta_max, df_err, fmax_per_image = \
                self._neb_compute_forces(
                    images_mw, energies, forces_mw, k_spring, self.masses,
                    frozen_indices=frozen_indices,
                )
    
            # 5c. Log
            fmax_history.append(fmax)
            if verbose:
                forces_spring_flat = np.concatenate(
                    [neb_forces_spring[i].flatten() for i in movable
                    if i in neb_forces_spring]
                )
                fpar = float(np.max(np.abs(forces_spring_flat))) \
                    if len(forces_spring_flat) > 0 else 0.0
                e_ts_display = np.max(energies)

                max_img_idx = max(fmax_per_image, key=fmax_per_image.get) if fmax_per_image else 0

                # Commented to add the diagnostics
                #print(f"{step:5d} {e_ts_display:14.8f} {fmax:12.6f} {frms:12.6f} {fpar:12.6f}")
                # no image
                #print(f"{step:5d} {e_ts_display:14.8f} {fmax:12.6f} {frms:12.6f} {fpar:12.6f} {theta_max:10.4f} {df_err:10.6f}")
                # image
                print(f"{step:5d} {e_ts_display:14.8f} {fmax:12.6f} {frms:12.6f} {fpar:12.6f} {max_img_idx:8d} {theta_max:10.4f}" )#$ {df_err:10.6f}")

            # 5d. Convergence
            if fmax < fmax_tol and frms < frms_tol:
                neb_converged = True
                if verbose:
                    print(f"\n  *** CONVERGED at step {step} ***")
                    print(f"  max|F_perp| = {fmax:.6f}  "
                        f"rms|F_perp| = {frms:.6f}")
                break
    
            # 5e. Optimizer step 
            if split_forces and optimizer in ('vpo', 'fire'):
                # SPLIT MODE: F_perp goes to optimizer while F_spring = steepest descent
                forces_perp_flat = np.concatenate(
                    [neb_forces_perp[i].flatten() for i in movable]
                )
                if precondition and m_precond is not None:
                    forces_perp_flat = forces_perp_flat * m_precond
    
                positions = np.concatenate(
                    [images_mw[i].flatten() for i in movable]
                )
    
                if scope == 'global':
                    new_pos = optimizers._neb_vpo_step(
                        positions, forces_perp_flat, opt_state,
                        dt=0.5, max_move=effective_max_move,
                    ) if optimizer == 'vpo' else optimizers._neb_fire_step(
                        positions, forces_perp_flat, opt_state,
                        max_move=effective_max_move,
                    )
                    step_opt = new_pos - positions
                else:
                    # Local: per-image
                    chunk = n_atoms * 3
                    step_opt = np.zeros_like(positions)
                    for j, i in enumerate(movable):
                        p_i = positions[j*chunk:(j+1)*chunk]
                        f_i = forces_perp_flat[j*chunk:(j+1)*chunk]
                        if optimizer == 'vpo':
                            new_p_i = optimizers._neb_vpo_step(
                                p_i, f_i, opt_state[i],
                                dt=0.5, max_move=effective_max_move,
                            )
                        else:
                            new_p_i = optimizers._neb_fire_step(
                                p_i, f_i, opt_state[i],
                                max_move=effective_max_move,
                            )
                        step_opt[j*chunk:(j+1)*chunk] = new_p_i - p_i
    
                # Spring: gentle steepest descent
                forces_spring_flat = np.concatenate(
                    [neb_forces_spring[i].flatten() for i in movable]
                )
                spring_step_size = 0.25
                step_spring = forces_spring_flat * spring_step_size
                spring_max = float(np.max(np.abs(step_spring)))
                if spring_max > 0.5 * effective_max_move:
                    step_spring *= 0.5 * effective_max_move / spring_max
    
                new_positions = positions + step_opt + step_spring
    
            else:
                # COMBINED MODE: optimizer handles all
                neb_forces_combined = {}
                for i in movable:
                    neb_forces_combined[i] = (
                        neb_forces_perp[i] + neb_forces_spring[i]
                    )
    
                if optimizer == 'lbfgs':
                    # Always per-image, updates images_mw directly
                    positions = np.concatenate(
                        [images_mw[i].flatten() for i in movable]
                    )
                    chunk = n_atoms * 3
                    new_positions = np.empty_like(positions)
                    for j, i in enumerate(movable):
                        pos_i = images_mw[i].flatten()
                        f_i = neb_forces_combined[i].flatten()
                        new_positions[j*chunk:(j+1)*chunk] = \
                            optimizers._neb_lbfgs_step_per_image(
                                pos_i, f_i, opt_state[i],
                                mem=lbfgs_mem, dr=lbfgs_dr,
                                max_move=effective_max_move,
                            )
    
                elif scope == 'global':
                    # Global VPO or FIRE, combined mode
                    positions = np.concatenate(
                        [images_mw[i].flatten() for i in movable]
                    )
                    forces_flat = np.concatenate(
                        [neb_forces_combined[i].flatten() for i in movable]
                    )
                    if optimizer == 'vpo':
                        if orca_vpo == False:
                            new_positions = optimizers._neb_vpo_step(
                                positions, forces_flat, opt_state,
                                dt=0.5, max_move=effective_max_move,
                            )
                        else:
                            new_positions = optimizers._neb_vpo_step_orca(
                            positions, forces_flat, opt_state,
                            dt=1.0, stepsize=orca_vpo_stepsize, max_move=effective_max_move,
                        )
                    else:
                        new_positions = optimizers._neb_fire_step(
                            positions, forces_flat, opt_state,
                            max_move=effective_max_move,
                        )
    
                else:
                    # Local VPO or FIRE, combined mode
                    positions = np.concatenate(
                        [images_mw[i].flatten() for i in movable]
                    )
                    forces_flat = np.concatenate(
                        [neb_forces_combined[i].flatten() for i in movable]
                    )
                    chunk = n_atoms * 3
                    new_positions = np.empty_like(positions)
                    for j, i in enumerate(movable):
                        p_i = positions[j*chunk:(j+1)*chunk]
                        f_i = forces_flat[j*chunk:(j+1)*chunk]
                        if optimizer == 'vpo':
                            new_positions[j*chunk:(j+1)*chunk] = \
                                optimizers._neb_vpo_step(
                                    p_i, f_i, opt_state[i],
                                    dt=0.5, max_move=effective_max_move,
                                )
                        else:
                            new_positions[j*chunk:(j+1)*chunk] = \
                                optimizers._neb_fire_step(
                                    p_i, f_i, opt_state[i],
                                    max_move=effective_max_move,
                                )
    
            # Unflatten back into images_mw
            chunk = n_atoms * 3
            for j, i in enumerate(movable):
                images_mw[i] = new_positions[
                    j * chunk : (j + 1) * chunk
                ].reshape(n_atoms, 3)
    
            # 5f. Save periodically (ideally at every step, but the user can change)
            if save_every > 0 and step % save_every == 0:
                neb_io._neb_save_state(
                    scratch_dir=scratch_dir, 
                    images_mw=images_mw, 
                    energies=energies, 
                    fmax_history=fmax_history, 
                    step=step, 
                    masses=self.masses,
                    atomic_numbers=self.atomic_numbers,
                    fmax_per_image=fmax_per_image,
                    job_prefix=job_prefix
                )

        else:
            if verbose:
                print(f"\n  WARNING: Did not converge in {max_steps} "
                    f"steps (fmax = {fmax:.6f}, frms = {frms:.6f})")
    
        # =============================================================
        #  6. Store results
        # =============================================================
        final_cart = np.array([mw / sqrt_m for mw in images_mw])
        self.raw_coord         = final_cart
        self.raw_coord_ang     = final_cart * const.bohr_to_ang
        self.mass_scaled_coord = np.array(images_mw)
        self.energies          = energies
        self.neb_converged     = neb_converged
        self.neb_fmax_history  = fmax_history
        self.neb_ts_idx        = ts_idx
        self.neb_frozen_indices = frozen_indices
    
        if hasattr(self, 'neb_images_mw'):
            self.neb_images_mw = None
    
        self._neb_save_state(
            scratch_dir=scratch_dir, 
            images_mw=images_mw, 
            energies=energies, 
            fmax_history=fmax_history, 
            step=step, 
            masses=self.masses,
            atomic_numbers=self.atomic_numbers,
            fmax_per_image=fmax_per_image,
            job_prefix=job_prefix
        )

        if verbose:
            if ts_idx is not None:
                bh = (energies[ts_idx] - energies[0]) * const.hartree_to_kcalmol
                print(f"Final barrier: {bh:.4f} kcal/mol (TS at idx {ts_idx})")
            else:
                e_max = np.max(energies)
                bh = (e_max - energies[0]) * const.hartree_to_kcalmol
                print(f"Final barrier: {bh:.4f} kcal/mol "
                    f"(highest image: {np.argmax(energies)})")
            print(f"Results in self.raw_coord, self.mass_scaled_coord, "
                f"self.energies.")