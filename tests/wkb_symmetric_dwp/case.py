"""
WKB semiclassical tunneling in symmetric DWP.

Covers:
  calc_tunneling        running calculate_relevant_vib, calculate_zpe,
                        integrate_action_integral and calc_tunneling_frequency_dwp_wkb_sym
  calc_half_life

"""


JSON_FILE      = 'tropolone_mwneb.json'
DWP            = True     # selects which tunneling-frequency route is taken

# TOLERANCES = {}


def run(API):
    tun = API()

    path = tun.load_cluster_results(json_file=JSON_FILE, verbose=False)
    path = tun.build_reaction_path(verbose=False)

    results = tun.calc_tunneling_wkb_sym(verbose=False)

    # tun.calc_half_life(verbose=False)

    return {
        # tunneling relatedc
        'tunneling_frequency': results.tunneling_frequency,
        'half_life':           results.half_life,

        # from integrate_action_integral
        'action_integral':     results.action_integral,
        'action_error':        results.action_integral_error,
        'roots':               results.roots,
        'V_minus_E':           results.V_minus_E,
        'sqroot':              results.sqroot,
        'positive_rp':         results.positive_rp,
        'wkb_energies':        results.wkb_energies, # it may come from different places, possible issue [TODO]

        # turning points and vibratons
        'relevant_vib':        results.vib.winner_freq,
        'turning_point':       results.turning_point,
    }

# Not saving interpol_VmE because it is a spline object, not serialisable for json
