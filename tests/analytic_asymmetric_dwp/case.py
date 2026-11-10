"""
Analytic asymmetric double well.

Covers:
  calc_asymmetric_energy_levels   running _unitless_var_calc,
                                  solve_asymmetric_finite_difference and
                                  calculate_level_coherences
"""

JSON_FILE    = 'hydrotropolone_mwneb.json'
INVERT_CURVE = True # it is true for the hydroxy tropolone

HI, BI, AI, LI = 10.0, 100.0, 0.0, 1        # initial guesses
N_ITER       = 100
L_STEP       = 0.1
RED_MASS     = 1.0
N_LEVELS     = 12


# The fitted parameters are obtained from an optimization, so the tolerance
# may need to be changed in case there are differenes in versions or some other
# small alteration.
# TOLERANCES = {'L': 1e-10}  # curve_fit output, may be sensitive to input last bit


def run(API):
    tun = API()

    tun.load_cluster_results(json_file=JSON_FILE,
                             invert_curve=INVERT_CURVE,
                             verbose=False)

    tun.build_reaction_path(verbose=False)

    results = tun.calc_asymmetric_energy_levels(hi=HI, bi=BI, ai=AI, Li=LI,
                                      n_iter=N_ITER,
                                      L_step=L_STEP,
                                      red_mass=RED_MASS,
                                      n_levels=N_LEVELS,
                                      verbose=False,
                                      verbose_L_fitting=False)

    # total_rate_s, half_life_s = tun.calculate_matrix_tunneling_rate(verbose=False)
    results = tun.calculate_matrix_tunneling_rate(verbose=False)

    return {
        'matrix_total_rate': results.total_rate_s,
        'matrix_half_life':  results.half_life_s,

        'tunneling_frequency': results.tunneling_frequency,
        'energy_levels':       results.energy_levels,
        'che_energy_levels':   results.che_energy_levels,
        'coherences':          results.coherences,

        'h_vec': results.h_vec,
        'b_vec': results.b_vec,
        'a_vec': results.a_vec,
        'L_vec': results.L_vec,
    }
