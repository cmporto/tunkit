"""
Analytic symmetric double well (APS).

Covers:
  calc_symmetric_energy_levels   running IRC_L_fit,
                                 _calculate_potential_parameters_stable,
                                 solve_oblate_spectral, and
                                 calculate_symmetric_level_coherences

Called straight after build_reaction_path, without calc_tunneling [TODO]
"""


JSON_FILE    = 'tropolone_mwneb.json'

RED_MASS     = 1.0
N_ITER       = 30
M_STEP       = 300
N_LEVELS     = 8
PREC_DIGITS  = 60 # not using mpmath yet
MATRIX_SIZE  = 1000
MI, HI, BI, LI = 1.0, 1.0, -41.0, 0.8e-10 # initial guesses
USE_MPMATH   = False


# The fitted parameters are obtained from an optimization, so the tolerance
# may need to be changed in case there are differenes in versions or some other
# small alteration.
# TOLERANCES = {'L': 1e-10}  # curve_fit output, may be sensitive to input last bit


def run(API):
    tun = API()

    tun.load_cluster_results(json_file=JSON_FILE,
                             # invert_curve=INVERT_CURVE,
                             verbose=False)

    tun.build_reaction_path(verbose=False)

    results = tun.calc_symmetric_energy_levels(n_iter=N_ITER,
                                     m_step=M_STEP,
                                     n_levels=N_LEVELS,
                                     prec_digits=PREC_DIGITS,
                                     mi=MI, hi=HI, bi=BI, Li=LI,
                                     matrix_size=MATRIX_SIZE,
                                     use_mpmath=USE_MPMATH,
                                     verbose=False,
                                     verbose_L_fitting=False)


    return {
        # tunneling results
        'tunneling_frequency': results.tunneling_frequency,
        'energy_levels':       results.energy_levels,
        'aps_energy_levels':   results.aps_energy_levels,
        'kinetic_rate':        results.kinetic_rate,

        # convergence history, interesting for maintainance but could possibly
        # need a looser convertgence
        'h_vec': results.h_vec,
        'b_vec': results.b_vec,
        'L_vec': results.L_vec,
        'm_vec': results.m_vec,
    }
