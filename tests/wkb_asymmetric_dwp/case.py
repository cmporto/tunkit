"""
WKB asymmetric tunneling in DWP (Rastelli 2012).

Covers:
  calc_tunneling_wkb_asym   running calculate_relevant_vib,
                            integrate_action_integral and _reverse_path_data

"""


JSON_FILE       = 'hydrotropolone_mwneb.json'
THRESHOLD       = 1.0E-4
INVERT_CURVE    = True

RMS_THRESHOLD   = 1e-4

# TOLERANCES = {}


def run(API):
    tun = API()

    tun.load_cluster_results(json_file=JSON_FILE,
                             invert_curve=INVERT_CURVE,
                             verbose=False)
    
    path = tun.build_reaction_path(verbose=False)
    
    # Better to do a separate path transformation folder
    raw_coord_ang_before = path.raw_coord_ang.copy()
    path_before = path.reaction_path.copy()
    proj_before = path.proj_energies.copy()

    result = tun.calc_tunneling_wkb_asym(rms_threshold=RMS_THRESHOLD,
                                verbose=False)

    return {
        # the answers
        'tunneling_frequency': result.tunneling_frequency,
        'level_splitting':     result.level_splitting,
        'half_life':           result.half_life,
        'tun_amplitude':       result.tun_amplitude,
        'relevant_vib':        result.reactant_vib.winner_freq,


        # before / after, pinning the in-place mutation done by
        # _reverse_path_data
        'path_before':                 path_before,
        'proj_before':                 proj_before,
        'path_after':                  path.reaction_path,
        'proj_after':                  path.proj_energies,
        'raw_coord_ang_before':        raw_coord_ang_before,
        'raw_coord_ang_after':         path.raw_coord_ang,
        'cartesian_disp_after':        path.cartesian_disp_non_mw,
        'displacement_per_step_after': path.displacement_per_step,
        'displacement_non_mw_after':   path.displacement_non_mw,
        'center_of_mass_after':        path.center_of_mass,
        'rotational_matrices_after':   path.rotational_matrices,
        'transl_vectors_after':        path.transl_vectors,
        'alignement_rmsd_after':       path.alignement_rmsd,
    }
