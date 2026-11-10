"""
Gaussian IRC extraction.

Covers: extract_coord, extract_energies, check_atomic_numbers, get_masses.

Folder is flat filenames resolve without any path handling.
"""

# Add tolerances when necessary. Everthing else is compared bitwise.
TOLERANCES = {'reaction_path': 1e-3}   # curve_fit output, may be sensitive to input last bit


def run(API):

    out ={}

    tun = API(mep_file_name='irc.log')

    tun.extract_irc_gaussian(verbose=False)
    
    tun.build_reaction_path()

    # tun2 = API()
    #
    # tun2.extract_irc(
    #     two_irc_files=True,
    #     first_irc_file_name='irc_fwd.log',
    #     second_irc_file_name='irc_rev.log',
    #     verbose=False
    #     )
    # tun2.build_reaction_path()

    out['energies']              = tun.path.energies
    out['raw_coord']             = tun.path.raw_coord
    out['raw_coord_ang']         = tun.path.raw_coord_ang
    out['atomic_numbers']        = tun.path.atomic_numbers
    out['masses']                = tun.path.masses
    out['reaction_path']         = tun.path.reaction_path
    # out['energies_2files']       = tun2.energies
    # out['raw_coord_2files']      = tun2.raw_coord
    # out['raw_coord_ang_2files']  = tun2.raw_coord_ang
    # out['atomic_numbers_2files'] = tun2.atomic_numbers
    # out['masses_2files']         = tun2.masses
    # out['reaction_path_2files']  = tun2.reaction_path

    return out
