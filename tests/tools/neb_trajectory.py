#!/usr/bin/env python3
"""
Developer script: save a finished MWNEB run as a test that runs without ORCA.

Run it on the folder of a saved run,

    <run_dir>/frozen/        image_000.*  and  image_<N-1>.*   endpoints, computed once
    <run_dir>/current_000/   image_001.* ... image_<N-2>.*     step 0
    <run_dir>/current_001/   ...                               step 1, and so on

and it writes, in <test_dir>:

    <name>.json   the molecule, the run_neb options of the run, and every
                  calculation (geometry from the .inp, energy and gradient from
                  the .engrad), read with orca.read_input_xyz and
                  orca.orca_engrad_parser
    golden.json   fmax and frms of every step, computed with
                  forces._neb_compute_forces on the recorded bands, plus the
                  number of steps and whether the run converged
    case.py       the test: runner.run_neb_from_json(<name>.json)

Then it runs the test once and compares it with golden.json, so run_neb
options that differ from the original run show up straight away.

Usage:
    python neb_trajectory.py <run_dir> <test_dir> --name ammonia --optimizer vpo --k-spring 0.1
    python neb_trajectory.py -h          (all the options)
"""

import argparse
import json
import os

import numpy as np

from tunkit.core import elements as elem
from tunkit.io import orca
from tunkit.neb import forces, runner

CASE_TEMPLATE = '''"""
Re-run of a saved MWNEB run: __NAME__.

__NAME__.json holds the molecule, the run_neb options and every calculation
of the original run. runner.run_neb_from_json starts from the same band and
reads the energies and gradients from that file; it stops with an error if
at some step the run asks for a geometry more than TOL (bohr) away from the
recorded ones. golden.json holds the fmax and frms of every step, computed
from the recorded run, the number of steps and whether it converged.

Written by tests/tools/neb_trajectory.py.
"""
import numpy as np

from tunkit.neb import runner

RUN_FILE = '__NAME__.json'
TOL = __TOL__        # bohr

TOLERANCES = {'fmax_history': 1e-8, 'frms_history': 1e-8}


def run(API):
    result = runner.run_neb_from_json(RUN_FILE, tol=TOL)
    return {'fmax_history': np.array(result.fmax_history),
            'frms_history': np.array(result.frms_history),
            'n_steps': result.n_steps,
            'converged': result.converged}
'''

CASE_TOLERANCES = {'fmax_history': 1e-8, 'frms_history': 1e-8}








def read_image(folder, i):
    """Geometry (bohr), energy and gradient of image i in a calculation folder."""
    stem = os.path.join(folder, f'image_{i:03d}')
    atomic_numbers, coords = orca.read_input_xyz(stem + '.inp')
    gradient, energy = orca.orca_engrad_parser(stem + '.engrad')
    return atomic_numbers, coords, energy, np.array(gradient).reshape(coords.shape)


def read_run(run_dir):
    """Endpoints, steps and the flat list of calculations of a saved run."""
    frozen = os.path.join(run_dir, 'frozen')
    ends = sorted(int(f[6:9]) for f in os.listdir(frozen) if f.endswith('.inp'))
    n_images = ends[-1] + 1
    movable = list(range(1, n_images - 1))

    atomic_numbers, R, E_R, G_R = read_image(frozen, 0)
    _, P, E_P, G_P = read_image(frozen, n_images - 1)
    calcs = {'image': [0, n_images - 1], 'step': [-1, -1], 'coords': [R, P],
             'energy': [E_R, E_P], 'gradient': [G_R, G_P]}

    steps = []
    k = 0
    while os.path.isdir(os.path.join(run_dir, f'current_{k:03d}')):
        folder = os.path.join(run_dir, f'current_{k:03d}')
        step = {'coords': [], 'energies': [], 'gradients': []}
        for i in movable:
            _, c, e, g = read_image(folder, i)
            for key, val in (('image', i), ('step', k), ('coords', c), ('energy', e), ('gradient', g)):
                calcs[key].append(val)
            step['coords'].append(c)
            step['energies'].append(e)
            step['gradients'].append(g)
        steps.append({key: np.array(val) for key, val in step.items()})
        k += 1

    return dict(atomic_numbers=atomic_numbers, n_images=n_images,
                endpoints=dict(coords=(R, P), energies=(E_R, E_P), gradients=(G_R, G_P)),
                steps=steps, calculations={key: np.array(val) for key, val in calcs.items()})


def golden_from_run(run, masses, settings):
    """fmax/frms of every recorded step, with the program's own force decomposition."""
    sqrt_m = np.sqrt(masses)[:, np.newaxis]
    n = run['n_images']
    R, P = run['endpoints']['coords']
    E_R, E_P = run['endpoints']['energies']
    F_R, F_P = [-g / sqrt_m for g in run['endpoints']['gradients']]

    fmax_hist, frms_hist = [], []
    for step in run['steps']:
        band_mw = [R * sqrt_m] + [c * sqrt_m for c in step['coords']] + [P * sqrt_m]
        band_mw = runner._align_product(band_mw, masses)   # as run_neb does before each step
        energies = np.concatenate([[E_R], step['energies'], [E_P]])
        forces_mw = [F_R] + [-g / sqrt_m for g in step['gradients']] + [F_P]
        _, _, fmax, frms, _, _, _ = forces._neb_compute_forces(
            band_mw, energies, forces_mw, settings['k_spring'], masses,
            frozen_indices={0, n - 1})
        fmax_hist.append(fmax)
        frms_hist.append(frms)

    converged = fmax_hist[-1] < settings['fmax_tol'] and frms_hist[-1] < settings['frms_tol']
    return {'fmax_history': fmax_hist, 'frms_history': frms_hist,
            'n_steps': len(run['steps']), 'converged': bool(converged)}








def save_test(run_dir, test_dir, name, settings, masses=None, tol=1e-7):
    run = read_run(run_dir)
    if masses is None:
        masses = elem.get_masses(run['atomic_numbers'])
    masses = np.asarray(masses, dtype=float)
    R, P = run['endpoints']['coords']
    initial_band = np.concatenate([[R], run['steps'][0]['coords'], [P]])

    record = {
        'description': f"MWNEB run saved from {os.path.abspath(run_dir)}",
        'atomic_numbers': run['atomic_numbers'].tolist(),
        'masses': masses.tolist(),
        'run_neb': dict(settings,
                        initial_band=initial_band.tolist(),
                        align_initial_band=False,
                        max_steps=len(run['steps'])),
        'calculations': {key: val.tolist() for key, val in run['calculations'].items()},
    }
    golden = golden_from_run(run, masses, settings)

    os.makedirs(test_dir, exist_ok=True)
    json_file = os.path.join(test_dir, f'{name}.json')
    with open(json_file, 'w') as fo:
        json.dump(record, fo)
    with open(os.path.join(test_dir, 'golden.json'), 'w') as fo:
        json.dump(golden, fo, indent=1, sort_keys=True)
    with open(os.path.join(test_dir, 'case.py'), 'w') as fo:
        fo.write(CASE_TEMPLATE.replace('__NAME__', name).replace('__TOL__', repr(tol)))
    print(f"{run['n_images']} images, {len(masses)} atoms, {len(run['steps'])} steps "
          f"-> {name}.json, golden.json, case.py in {test_dir}")

    # one run of the new test, so wrong options show up now
    try:
        result = runner.run_neb_from_json(json_file, tol=tol)
    except ValueError as e:
        print(f"CHECK FAILED: {e}\n  Are the options the same as in the original run?")
        return False
    problems = []
    for key, atol in CASE_TOLERANCES.items():
        got = np.array(getattr(result, key))
        exp = np.array(golden[key])
        if got.shape != exp.shape or not np.allclose(got, exp, rtol=0, atol=atol):
            problems.append(key)
    if result.n_steps != golden['n_steps']:
        problems.append(f"n_steps {result.n_steps} vs {golden['n_steps']}")
    if result.converged != golden['converged']:
        problems.append(f"converged {result.converged} vs {golden['converged']}")
    if problems:
        print(f"CHECK FAILED: {problems}. Are the options the same as in the original run?")
        return False
    print(f"Check: the re-run matches the saved run "
          f"({golden['n_steps']} steps, converged={golden['converged']}).")
    return True








def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('run_dir')
    ap.add_argument('test_dir')
    ap.add_argument('--name', default=None, help='JSON file name (default: the test folder name)')
    ap.add_argument('--k-spring', type=float, default=0.1)
    ap.add_argument('--optimizer', choices=['vpo', 'fire', 'lbfgs'], default='vpo')
    ap.add_argument('--scope', choices=['global', 'local'], default='global')
    ap.add_argument('--split-forces', action='store_true')
    ap.add_argument('--precondition', action='store_true')
    ap.add_argument('--max-move', type=float, default=0.2)
    ap.add_argument('--lbfgs-mem', type=int, default=20)
    ap.add_argument('--lbfgs-dr', type=float, default=1.0e-3)
    ap.add_argument('--align-every-step', action='store_true')
    ap.add_argument('--fmax-tol', type=float, default=4.5e-4)
    ap.add_argument('--frms-tol', type=float, default=3.0e-4)
    ap.add_argument('--masses', default=None,
                    help='comma-separated masses in amu (isotopes); default: standard')
    ap.add_argument('--tol', type=float, default=1e-7,
                    help='largest geometry difference (bohr) accepted at any step')
    a = ap.parse_args()

    settings = dict(k_spring=a.k_spring, optimizer=a.optimizer, scope=a.scope,
                    split_forces=a.split_forces, precondition=a.precondition,
                    max_move=a.max_move, lbfgs_mem=a.lbfgs_mem, lbfgs_dr=a.lbfgs_dr,
                    align_every_step=a.align_every_step,
                    fmax_tol=a.fmax_tol, frms_tol=a.frms_tol)
    masses = [float(x) for x in a.masses.split(',')] if a.masses else None
    name = a.name or os.path.basename(os.path.normpath(a.test_dir))
    ok = save_test(a.run_dir, a.test_dir, name, settings, masses=masses, tol=a.tol)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
