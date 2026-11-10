"""
Gaussian projection.

Covers:
  extract_projection_energies   (used before the payload. Multiple files and larger
                                 disk usage. Reference for future payload implement)
  parse_gaussian_archive        (pure parser, tested directly on one file)

Data: the folder of Gaussian .log files for the points along the path.
"""


ARCHIVE_FILE = 'tropolone_p011.log'     # any single .log, for the direct parser test


# Add tolerances when necessary. Everthing else is compared bitwise.
# TOLERANCES = {'L': 1e-10}   # curve_fit output, may be sensitive to input last bit
TOLERANCES = {'energies': 1e-7}

def run(API):
    out = {} 

    # test runs over a chain of methods, namely:
    # parse_gaussian_archive, get_masses, align_coordinates, project_frequencies,
    # calculate_zpe, _project_out_tr
    # diagnostic_path_and_proj_freq is checked in another test folder
    tun = API()
    path = tun.extract_projection_energies(path_to_folder='./projection_files', verbose=False)

    out['energies']              = path.energies
    out['proj_energies']         = path.proj_energies
    out['projected_corrections'] = path.projected_corrections
    out['path_frequencies']      = path.path_frequencies
    out['proj_coordinates']      = path.proj_coordinates
    out['raw_coord']             = path.raw_coord
    out['atomic_numbers']        = path.atomic_numbers
    out['masses']                = path.masses
    out['mw_hessians']           = path.mw_hessians

    # Deliberately not stored: hessians, mw_hessians, gradients, mw_gradients,
    # normal_modes. Large, and covered transitively -- path_frequencies cannot
    # match bitwise unless the Hessians did.

    # Checking the parser alone (helps isolate problems...)
    from tunkit.io.gaussian import parse_gaussian_archive
    coords, hessian, gradient, energy, atomic_numbers = \
        parse_gaussian_archive(ARCHIVE_FILE)

    out['archive_coords']         = coords
    out['archive_gradient']       = gradient
    out['archive_energy']         = energy
    out['archive_atomic_numbers'] = atomic_numbers
    out['archive_hessian_trace']  = hessian #.trace()   # cheaper?

    return out
