"""
MWNEB from real run data.

Folder contents:
    case.py
    restart.json          the last (converged) band, from the end of run_neb
    force_scratch/        .engrad for every image of the band in the last band
                          before convergence, i.e. the second to last band

Covers:
    orca_engrad_parser        sorted gradients
    _neb_align_band           re-alignment of an already-aligned band
    _neb_compute_tangent      every interior image
    _neb_compute_forces       the real NEB force decomposition
    _neb_vpo_step             one step from the real band and real forces
    _neb_read_restart         the real converged band
    _neb_save_state           round trip: save the band, read it back
    _neb_resample_path        spline resampling of the real band

"""

import os
import shutil
import tempfile

import numpy as np
from tunkit.neb import neb_io
from tunkit.neb import optimizers
from tunkit.neb import forces
from tunkit.neb import band
from tunkit.io import orca

RESTART_FILE     = 'restart.json'
FORCE_DIR        = 'force_scratch'
SEC_TO_LAST_BAND = 'carb_acid_mwneb_second_to_last_band.xyz'

K_SPRING      = 0.1        # k_spring used in the run
DT            = 0.5        # VPO dt, matches the run_neb call site
MAX_MOVE      = 0.2        # effective_max_move used in the run
N_RESAMPLE    = 90         # tested file has 80 images


# TOLERANCES = {}


def _read_gradients(tun, dirpath):
    """
    Gradients in the same order _neb_read_directory used: .hess files sorted
    alphabetically, each paired with <stem>.engrad. Mirroring that sort is
    what keeps image i's gradient with image i's geometry.
    """
    engrad_files = sorted(f for f in os.listdir(dirpath) if f.endswith('.engrad'))
    gradients, energies = [], []
    for efile in engrad_files:
        gradient, energy = orca.orca_engrad_parser(
            os.path.join(dirpath, efile))
        gradients.append(np.asarray(gradient))
        energies.append(energy)
    return np.asarray(gradients), np.asarray(energies), engrad_files


def run(API):
    out = {}
    a = API(SEC_TO_LAST_BAND)

    path = a.extract_coord_energy_xyz(verbose=False)
    coords_prev = path.raw_coord
    # coords_prev = a._neb_read_directory(FORCE_DIR, verbose=False)
    # coords_prev = np.asarray(coords_prev)

    n_images = len(coords_prev)
    n_atoms  = len(path.masses)
    a.masses = path.masses
    a.atomic_numbers = path.atomic_numbers
    sqrt_m   = np.sqrt(path.masses)[:, np.newaxis]

    out['prev_coords']         = coords_prev
    out['prev_energies']       = path.energies
    out['prev_atomic_numbers'] = path.atomic_numbers
    out['prev_masses']         = path.masses

    gradients, engrad_energies, engrad_files = _read_gradients(a, FORCE_DIR)
    out['gradients']       = gradients
    out['engrad_energies'] = engrad_energies
    out['image_order']     = engrad_files      # pins the pairing itself

    # tangents
    images_mw = coords_prev * sqrt_m
    # energies_prev = np.asarray(path.energies, dtype=float)
    tangents = np.asarray([
        forces._neb_compute_tangent(path.energies, images_mw, i)
        for i in range(1, n_images - 1)
    ])
    out['tangents'] = tangents

    #  NEB forces
    # Sign and mass-weighting follow _neb_parse_results:
    #   f_cart = -gradient ;  forces_mw = f_cart / sqrt_m
    forces_mw = np.asarray([
        (-gradients[i].reshape(n_atoms, 3)) / sqrt_m for i in range(n_images)
    ])
    out['forces_mw'] = forces_mw

    frozen = [0, n_images - 1]
    (f_perp, f_spring, fmax, frms,
     theta_max, df_err_max, fmax_per_image) = forces._neb_compute_forces(
        images_mw, path.energies, forces_mw, K_SPRING, path.masses, frozen_indices=frozen)

    out['neb_forces_perp']   = np.asarray([f_perp[i] for i in sorted(f_perp)])
    out['neb_forces_spring'] = np.asarray([f_spring[i] for i in sorted(f_spring)])
    out['neb_fmax']          = fmax
    out['neb_frms']          = frms
    out['neb_theta_max']     = theta_max
    out['neb_df_err_max']    = df_err_max
    out['neb_fmax_per_image'] = np.asarray(
        [fmax_per_image[i] for i in sorted(fmax_per_image)])

    # one VPO step
    movable = list(range(1, n_images - 1))
    positions        = np.concatenate([images_mw[i].flatten() for i in movable])
    forces_perp_flat = np.concatenate([np.asarray(f_perp[i]).flatten()
                                       for i in movable])

    state = {}          # fresh as the real velocity is not yet saved
    new_pos = optimizers._neb_vpo_step(positions, forces_perp_flat, state,
                              dt=DT, max_move=MAX_MOVE)

    out['vpo_positions_in']  = positions
    out['vpo_positions_out'] = np.asarray(new_pos)
    out['vpo_step']          = np.asarray(new_pos) - positions
    out['vpo_velocity']      = np.asarray(state.get('v', []))

    # the converged band
    b = API()
    b.masses = path.masses
    b.atomic_numbers = path.atomic_numbers
    coords_final, energies = neb_io._neb_read_restart(RESTART_FILE, b.masses, verbose=False)


    out['final_coords']   = coords_final
    out['final_energies'] = energies

    final_mw = coords_final * sqrt_m

    # Not expected to be zero as the velocity is not yeat saved
    # final_movable = np.concatenate([final_mw[i].flatten() for i in movable])
    # out['step_residual']      = np.asarray(new_pos) - final_movable
    # out['step_residual_norm'] = float(np.linalg.norm(
    #     np.asarray(new_pos) - final_movable))

    # save / read test
    scratch = tempfile.mkdtemp(prefix='neb_golden_')
    try:
        neb_io._neb_save_state(scratch_dir=scratch, 
                               images_mw=final_mw, 
                               energies=energies,
                               fmax_history=[0.0], 
                               step=0, 
                               masses=path.masses, 
                               atomic_numbers=path.atomic_numbers )

        c = API()
        c.masses = a.masses
        c.atomic_numbers = a.atomic_numbers
        roundtrip = np.asarray(neb_io._neb_read_restart(os.path.join(scratch, 'restart.json'), 
                                                   path.masses,
                                                   verbose=False)[0])
        # testing if save + read are the same
        out['roundtrip_residual'] = roundtrip - coords_final
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    # resampling
    resampled = band._neb_resample_path(list(coords_final), 
                                        N_RESAMPLE,
                                        path.masses)
    out['resampled']            = resampled
    out['resampled_first_delta'] = resampled[0] - coords_final[0]
    out['resampled_last_delta']  = resampled[-1] - coords_final[-1]

    return out
