"""
ORCA/NEB/MWNEB extraction and path diagnostics

Covers:
  extract_coord_energy_xyz         multi-frame xyz with energies in comments
  _neb_read_multi_xyz              the NEB-side reader of the same format
  read_orca_data                   .hess parser
  orca_engrad_parser               .engrad parser
  extract_projection_energies_neb  the full ORCA projection pipeline
  These are the four metrics to diagnose_mwneb_path_health:
  _calc_arc_lengths
  _calc_kinking_angles
  _calc_tangent_consistency
  _calc_gradient_overlaps

diagnose_mwneb_path_health only prints, so each diagnostic method is
checked by itself.

Data: NEB run output
"""

import numpy as np
from tunkit.path import diagnostics as diag
from tunkit.neb.neb_io import _neb_read_multi_xyz

PATH_XYZ  = 'neb_path_tropolone.xyz'      # multi-frame xyz for the whole band
HESS_FILE = '000.hess'     # any single .hess, for the direct parser test
ENGRAD_FILE = '000.engrad'  # matching .engrad

# Add tolerances when necessary. Everthing else is compared bitwise.
# TOLERANCES = {}    # curve_fit outputs may be sensitive to input last bit


def run(API):
    out = {}

    # Testing both parsers
    a = API(mep_file_name=PATH_XYZ)
    path = a.extract_coord_energy_xyz(verbose=False)

    out['xyz_energies']       = path.energies
    out['xyz_raw_coord']      = path.raw_coord
    out['xyz_raw_coord_ang']  = path.raw_coord_ang
    out['xyz_atomic_numbers'] = path.atomic_numbers
    out['xyz_masses']         = path.masses

    b = API()
    all_coords, energies, masses, atomic_numbers = _neb_read_multi_xyz(PATH_XYZ, verbose=False)

    out['multixyz_coord_bohr']     = all_coords #np.asarray(frames)
    out['multixyz_energies']       = energies #b.energies
    out['multixyz_atomic_numbers'] = atomic_numbers #b.atomic_numbers
    out['multixyz_masses']         = masses #b.masses

    # the two ORCA parsers on their own

    b2 = API()
    from tunkit.io.orca import read_orca_data, orca_engrad_parser
    masses, atoms, coords, hessian, normal_modes, freqs = read_orca_data(HESS_FILE)

    out['hess_masses']       = masses
    out['hess_atoms']        = atoms
    out['hess_coords']       = coords
    out['hess_frequencies']  = freqs
    out['hess_trace']        = np.asarray(hessian).trace()
    out['hess_nm_shape']     = list(np.asarray(normal_modes).shape)

    gradient, energy = orca_engrad_parser(ENGRAD_FILE)

    out['engrad_gradient'] = gradient
    out['engrad_energy']   = energy

    # the extract_projection_energies_neb
    c = API()
    path = c.extract_projection_energies_neb(path_to_folder='./projection_files', verbose=False)

    out['neb_energies']              = path.energies
    out['neb_proj_energies']         = path.proj_energies
    out['neb_projected_corrections'] = path.projected_corrections
    out['neb_path_frequencies']      = path.path_frequencies
    out['neb_projected_frequencies'] = path.projected_frequencies
    out['neb_proj_coordinates']      = path.proj_coordinates
    out['neb_tangents']              = path.tangents
    out['neb_masses']                = path.masses
    # if path frequencies and projected frequencies are correct, we can say the Hessians were too

    # path diagnostics
    # mass_scaled_coord comes from build_reaction_path that calls mass_weight_coordinates
    # mw_gradients the gradients (tangents) com from extract_projection_energies_neb
    path = c.build_reaction_path(verbose=False)

    out['reaction_path']     = path.reaction_path
    out['mass_scaled_coord'] = path.mass_scaled_coord

    out['diag_arc_lengths']        = diag._calc_arc_lengths(path.mass_scaled_coord)
    out['diag_kinking_angles']     = diag._calc_kinking_angles(path.mass_scaled_coord)
    out['diag_tangent_shifts']     = diag._calc_tangent_consistency(path.mass_scaled_coord)
    out['diag_gradient_overlaps']  = diag._calc_gradient_overlaps(path.mw_gradients)

    return out
