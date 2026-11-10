"""
Re-run of a saved MWNEB run: ammonia.

ammonia.json holds the molecule, the run_neb options and every calculation
of the original run. runner.run_neb_from_json starts from the same band and
reads the energies and gradients from that file; it stops with an error if
at some step the run asks for a geometry more than TOL (bohr) away from the
recorded ones. golden.json holds the fmax and frms of every step, computed
from the recorded run, the number of steps and whether it converged.

Written by tests/tools/neb_trajectory.py.
"""
import numpy as np

from tunkit.neb import runner

RUN_FILE = 'ammonia.json'
TOL = 1e-07        # bohr

TOLERANCES = {'fmax_history': 1e-8, 'frms_history': 1e-8}


def run(API):
    result = runner.run_neb_from_json(RUN_FILE, tol=TOL)
    return {'fmax_history': np.array(result.fmax_history),
            'frms_history': np.array(result.frms_history),
            'n_steps': result.n_steps,
            'converged': result.converged}
