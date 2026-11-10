#!/usr/bin/env python3
"""
Turn a saved MWNEB run into a replay test, and measure a replay against it.

A saved run is a folder with

    frozen/        image_000.*  and  image_<N-1>.*   endpoints, computed once
    current_000/   image_001.* ... image_<N-2>.*     step 0
    current_001/   ...                               step 1
    ...

Each image needs its .inp (the geometry ORCA received) and its .engrad
(energy and gradient). Everything else (.out, .gbw, .hess...) is ignored.
Every step is needed, from step 0 and without gaps: the optimizer state
(VPO/FIRE velocity, L-BFGS history) at step k depends on all the forces
before it, and the replay rebuilds it step by step.

extract
    Reads the folders with the program's own parsers and writes, in the
    test folder:
      recorded_run.json  settings + per-step geometries, energies, gradients
      golden.json        the geometries evaluated at each step, and the
                         fmax/frms the program computes for them
                         (forces._neb_compute_forces)
      case.py            the stub that run_golden.py runs
    Then it replays once and prints the deviation table, so wrong settings
    (optimizer, k_spring, max_move...) show up straight away.

replay
    Runs the current runner on recorded_run.json with
    engines.recorded_steps_engrad, and prints per step how far the
    geometries and fmax are from the recorded run. Read the table down to
    the first flagged step: after it, the replay keeps going with recorded
    forces that belong to the recorded geometries, so later rows show how
    the difference propagates, not what a new run would do.

Usage
    python neb_trajectory.py extract <run_dir> <test_dir> [settings]
    python neb_trajectory.py replay  <test_dir>
    python neb_trajectory.py extract -h        (all the settings)
"""

import argparse
import functools
import json
import os
import re
import shutil
import sys
import tempfile

import numpy as np

from tunkit.core import elements as elem
from tunkit.io import orca
from tunkit.neb import engines, forces, neb_io, runner
from tunkit.path.reaction_path import ReactionPath

# Run options that change the trajectory. Same names and defaults as runner.run_neb.
SETTINGS_DEFAULTS = dict(k_spring=0.1, optimizer='vpo', scope='global',
                         split_forces=False, precondition=False, max_move=0.2,
                         lbfgs_mem=20, lbfgs_dr=1.0e-3, align_every_step=False,
                         fmax_tol=4.5e-4, frms_tol=3.0e-4)

# The saved .inp has 10 decimals in Angstrom, so a replay starts about 1e-10 bohr
# away from the original numbers. A behavior change shows up orders of magnitude
# above these.
TOLERANCES = {'step_coords': 1e-7, 'fmax_history': 1e-8, 'frms_history': 1e-8}

STEP_PATTERN = r'^current[_-]?(\d+)$'

CASE_STUB = '''"""
Replay of a saved MWNEB run (written by tools/neb_trajectory.py extract).

recorded_run.json  what the original run computed at every step
golden.json        the geometries it evaluated and the fmax/frms of each step

The current runner starts from the same band and gets the recorded energies
and gradients step by step (engines.recorded_steps_engrad). Matching the
recorded geometries at every step means the optimizer (velocities, L-BFGS
history, alignment) behaves as it did in the original run.

Per-step table of the deviations:
    python ../tools/neb_trajectory.py replay .
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tools'))
import neb_trajectory  # noqa: E402

TOLERANCES = neb_trajectory.TOLERANCES


def run(API):
    return neb_trajectory.run_case(HERE)
'''








# =============================================================
#  Reading the saved run
# =============================================================

def find_step_folders(run_dir, pattern=STEP_PATTERN):
    """Numbered step folders, sorted by number. Raises on gaps."""
    regex = re.compile(pattern)
    found = []
    for name in os.listdir(run_dir):
        m = regex.match(name)
        if m and os.path.isdir(os.path.join(run_dir, name)):
            found.append((int(m.group(1)), os.path.join(run_dir, name)))
    found.sort()
    if not found:
        raise FileNotFoundError(f"No step folders matching {pattern!r} in {run_dir}")
    numbers = [n for n, _ in found]
    expected = list(range(numbers[0], numbers[0] + len(numbers)))
    if numbers != expected:
        missing = sorted(set(expected) - set(numbers))
        raise ValueError(f"Step folders are not consecutive, missing {missing}. "
                         f"The replay needs every step.")
    if os.path.isdir(os.path.join(run_dir, 'current')):
        print("  Note: ignoring the unnumbered 'current' folder (the run's scratch).")
    return [path for _, path in found]


def read_folder(folder):
    """
    {image: (atomic_numbers, coords_bohr, energy, gradient)} for every image_XXX.inp
    in the folder, read with neb_io._neb_read_input_xyz and orca.orca_engrad_parser.
    """
    images = {}
    for name in sorted(os.listdir(folder)):
        m = re.match(r'^image_(\d+)\.inp$', name)
        if not m:
            continue
        i = int(m.group(1))
        engrad = os.path.join(folder, f'image_{i:03d}.engrad')
        if not os.path.exists(engrad):
            raise FileNotFoundError(f"{engrad} is missing (did ORCA fail for this image?)")
        atomic_numbers, coords = neb_io._neb_read_input_xyz(os.path.join(folder, name))
        gradient, energy = orca.orca_engrad_parser(engrad)
        gradient = np.array(gradient).reshape(len(atomic_numbers), 3)
        images[i] = (atomic_numbers, coords, energy, gradient)
    if not images:
        raise FileNotFoundError(f"No image_XXX.inp files in {folder}")
    return images


def _block(images):
    idx = sorted(images)
    return {'indices': idx,
            'coords': np.array([images[i][1] for i in idx]),
            'energies': np.array([images[i][2] for i in idx]),
            'gradients': np.array([images[i][3] for i in idx])}


def read_run(run_dir, frozen_dir='frozen', pattern=STEP_PATTERN, n_steps=None):
    """Endpoints and steps of a saved run, as arrays."""
    frozen = read_folder(os.path.join(run_dir, frozen_dir))
    if len(frozen) != 2 or min(frozen) != 0:
        raise ValueError(f"{frozen_dir}/ should hold image_000 and the last image, "
                         f"found images {sorted(frozen)}")
    n_images = max(frozen) + 1
    movable = list(range(1, n_images - 1))
    atomic_numbers = frozen[0][0]

    folders = find_step_folders(run_dir, pattern)
    if n_steps is not None:
        folders = folders[:n_steps]
    steps = []
    for k, folder in enumerate(folders):
        images = read_folder(folder)
        if sorted(images) != movable:
            raise ValueError(f"{os.path.basename(folder)} has images {sorted(images)}, "
                             f"expected {movable[0]}..{movable[-1]}")
        for i, (an, *_rest) in images.items():
            if not np.array_equal(an, atomic_numbers):
                raise ValueError(f"Atom order differs in {folder}, image {i}")
        steps.append(_block(images))
    return {'atomic_numbers': atomic_numbers, 'n_images': n_images,
            'endpoints': _block(frozen), 'steps': steps,
            'step_folders': [os.path.basename(f) for f in folders]}








# =============================================================
#  Golden values, computed with the program's functions
# =============================================================

def golden_from_recording(rec):
    """
    For every recorded step: the movable geometries, and fmax/frms from
    forces._neb_compute_forces on the recorded band (product re-aligned onto
    image N-2 with runner._align_product, as the runner does before each step).
    """
    masses = np.asarray(rec['masses'])
    sqrt_m = np.sqrt(masses)[:, np.newaxis]
    s = rec['settings']
    n = rec['n_images']
    R, P = rec['endpoints']['coords']
    E_R, E_P = rec['endpoints']['energies']
    F_R, F_P = [-g / sqrt_m for g in rec['endpoints']['gradients']]

    fmax_hist, frms_hist = [], []
    for step in rec['steps']:
        band_mw = [R * sqrt_m] + [c * sqrt_m for c in step['coords']] + [P * sqrt_m]
        band_mw = runner._align_product(band_mw, masses)
        energies = np.concatenate([[E_R], step['energies'], [E_P]])
        forces_mw = [F_R] + [-g / sqrt_m for g in step['gradients']] + [F_P]
        _, _, fmax, frms, _, _, _ = forces._neb_compute_forces(
            band_mw, energies, forces_mw, s['k_spring'], masses,
            frozen_indices={0, n - 1})
        fmax_hist.append(fmax)
        frms_hist.append(frms)

    conv = [f < s['fmax_tol'] and r < s['frms_tol'] for f, r in zip(fmax_hist, frms_hist)]
    early = [k for k, c in enumerate(conv[:-1]) if c]
    if early:
        print(f"  WARNING: by these tolerances the run had already converged at step "
              f"{early[0]}, but {len(conv)} steps were saved. Check fmax_tol/frms_tol.")
    return {'step_coords': np.array([st['coords'] for st in rec['steps']]),
            'fmax_history': np.array(fmax_hist),
            'frms_history': np.array(frms_hist),
            'n_steps': len(rec['steps']),
            'converged': bool(conv[-1])}








# =============================================================
#  Replay
# =============================================================

def replay(rec, verbose=False, **overrides):
    """
    Run the current runner on a recording. Returns (replayed, result) where
    replayed has the same keys as the golden values.
    """
    settings = dict(rec['settings'])
    settings.update(overrides)
    atomic_numbers = np.asarray(rec['atomic_numbers'])
    path = ReactionPath(masses=np.asarray(rec['masses']), atomic_numbers=atomic_numbers)
    calls = []
    engine = functools.partial(engines.recorded_steps_engrad, recording=rec, calls=calls)
    R, P = rec['endpoints']['coords']

    if rec['start'] == 'recorded':
        start = dict(initial_band=np.concatenate([[R], rec['steps'][0]['coords'], [P]]),
                     align_initial_band=False)
    elif rec['start'] == 'cold':
        start = dict(reactant_coords=R, product_coords=P, n_images=rec['n_images'])
    else:
        raise ValueError(f"Unknown start {rec['start']!r}")

    scratch = tempfile.mkdtemp(prefix='neb_replay_')
    try:
        result = runner.run_neb(path, engine=engine, max_steps=len(rec['steps']),
                                save_every=0, scratch_dir=scratch, verbose=verbose,
                                **start, **settings)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    replayed = {'step_coords': np.array([c['coords'] for c in calls[1:]]),
                'fmax_history': np.array(result.fmax_history),
                'frms_history': np.array(result.frms_history),
                'n_steps': result.n_steps,
                'converged': bool(result.converged)}
    return replayed, result


def deviation_table(golden, replayed, tol=TOLERANCES['step_coords']):
    """Print, per step, the largest geometry deviation and the fmax of both runs."""
    gc, rc = golden['step_coords'], replayed['step_coords']
    n = min(len(gc), len(rc))
    print(f"\n{'step':>5s} {'max|dx| (bohr)':>15s} {'image':>6s} "
          f"{'fmax recorded':>14s} {'fmax replay':>13s} {'|d fmax|':>10s}")
    print('-' * 68)
    first_bad = None
    worst = 0.0
    for k in range(n):
        d = np.abs(rc[k] - gc[k])
        dmax = float(d.max())
        img = int(np.unravel_index(d.argmax(), d.shape)[0]) + 1
        fr, fp = golden['fmax_history'][k], replayed['fmax_history'][k]
        flag = '  <--' if dmax > tol and first_bad is None else ''
        if dmax > tol and first_bad is None:
            first_bad = k
        worst = max(worst, dmax)
        print(f"{k:5d} {dmax:15.3e} {img:6d} {fr:14.6e} {fp:13.6e} {abs(fr - fp):10.2e}{flag}")
    print('-' * 68)
    print(f"steps: recorded {golden['n_steps']}, replayed {replayed['n_steps']} | "
          f"converged: recorded {golden['converged']}, replayed {replayed['converged']}")
    if first_bad is None and golden['n_steps'] == replayed['n_steps'] \
            and golden['converged'] == replayed['converged']:
        print(f"MATCH: largest deviation {worst:.2e} bohr (tolerance {tol:.0e}).")
    elif first_bad is not None:
        print(f"DIVERGED at step {first_bad}. If this is step 1 or 2 right after "
              f"extract, the settings probably differ from the original run.")
    else:
        print("MISMATCH in the number of steps or the convergence flag.")
    return first_bad








# =============================================================
#  Files
# =============================================================

def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def load_recording(test_dir):
    with open(os.path.join(test_dir, 'recorded_run.json')) as fo:
        rec = json.load(fo)
    for blk in [rec['endpoints']] + rec['steps']:
        for key in ('coords', 'energies', 'gradients'):
            blk[key] = np.array(blk[key], dtype=float)
    return rec


def load_golden(test_dir):
    with open(os.path.join(test_dir, 'golden.json')) as fo:
        g = json.load(fo)
    for key in ('step_coords', 'fmax_history', 'frms_history'):
        g[key] = np.array(g[key], dtype=float)
    return g


def extract(run_dir, test_dir, settings, masses=None, start='recorded',
            frozen_dir='frozen', pattern=STEP_PATTERN, n_steps=None):
    print(f"Reading {run_dir} ...")
    rec = read_run(run_dir, frozen_dir=frozen_dir, pattern=pattern, n_steps=n_steps)
    n_atoms = len(rec['atomic_numbers'])
    rec['masses'] = (np.asarray(masses, dtype=float) if masses is not None
                     else elem.get_masses(rec['atomic_numbers']))
    if len(rec['masses']) != n_atoms:
        raise ValueError(f"{len(rec['masses'])} masses for {n_atoms} atoms")
    rec['settings'] = dict(SETTINGS_DEFAULTS, **settings)
    rec['start'] = start
    print(f"  {rec['n_images']} images, {n_atoms} atoms, {len(rec['steps'])} steps "
          f"({rec['step_folders'][0]} .. {rec['step_folders'][-1]})")

    golden = golden_from_recording(rec)
    os.makedirs(test_dir, exist_ok=True)
    with open(os.path.join(test_dir, 'recorded_run.json'), 'w') as fo:
        json.dump(_jsonable(rec), fo)
    with open(os.path.join(test_dir, 'golden.json'), 'w') as fo:
        json.dump(_jsonable(golden), fo, indent=1, sort_keys=True)
    with open(os.path.join(test_dir, 'case.py'), 'w') as fo:
        fo.write(CASE_STUB)
    print(f"  wrote recorded_run.json, golden.json and case.py in {test_dir}")

    print("\nReplaying with the current runner to check the settings...")
    replayed, _ = replay(load_recording(test_dir))
    deviation_table(load_golden(test_dir), replayed)


def run_case(test_dir):
    """What case.py returns to run_golden.py."""
    replayed, _ = replay(load_recording(test_dir))
    return replayed








def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    ex = sub.add_parser('extract', help='saved run folder -> replay test folder')
    ex.add_argument('run_dir')
    ex.add_argument('test_dir')
    ex.add_argument('--k-spring', type=float, default=SETTINGS_DEFAULTS['k_spring'])
    ex.add_argument('--optimizer', choices=['vpo', 'fire', 'lbfgs'], default='vpo')
    ex.add_argument('--scope', choices=['global', 'local'], default='global')
    ex.add_argument('--split-forces', action='store_true')
    ex.add_argument('--precondition', action='store_true')
    ex.add_argument('--max-move', type=float, default=SETTINGS_DEFAULTS['max_move'])
    ex.add_argument('--lbfgs-mem', type=int, default=SETTINGS_DEFAULTS['lbfgs_mem'])
    ex.add_argument('--lbfgs-dr', type=float, default=SETTINGS_DEFAULTS['lbfgs_dr'])
    ex.add_argument('--align-every-step', action='store_true')
    ex.add_argument('--fmax-tol', type=float, default=SETTINGS_DEFAULTS['fmax_tol'])
    ex.add_argument('--frms-tol', type=float, default=SETTINGS_DEFAULTS['frms_tol'])
    ex.add_argument('--masses', default=None,
                    help='comma-separated masses in amu (isotopes); default: standard')
    ex.add_argument('--start', choices=['recorded', 'cold'], default='recorded',
                    help="'recorded': start from the step-0 band (any run). "
                         "'cold': re-interpolate between the endpoints (cold-start runs)")
    ex.add_argument('--steps', type=int, default=None,
                    help='keep only the first N steps (smaller test for big systems)')
    ex.add_argument('--frozen-dir', default='frozen')
    ex.add_argument('--step-pattern', default=STEP_PATTERN,
                    help=f'regex for step folder names, group 1 = step number '
                         f'(default {STEP_PATTERN!r})')

    rp = sub.add_parser('replay', help='replay a test folder and print the deviations')
    rp.add_argument('test_dir')

    args = ap.parse_args()
    try:
        _dispatch(args)
    except (ValueError, FileNotFoundError, IndexError) as e:
        print(f"ERROR: {e}")
        return 1
    return 0


def _dispatch(args):
    if args.cmd == 'extract':
        settings = dict(k_spring=args.k_spring, optimizer=args.optimizer, scope=args.scope,
                        split_forces=args.split_forces, precondition=args.precondition,
                        max_move=args.max_move, lbfgs_mem=args.lbfgs_mem,
                        lbfgs_dr=args.lbfgs_dr, align_every_step=args.align_every_step,
                        fmax_tol=args.fmax_tol, frms_tol=args.frms_tol)
        masses = [float(x) for x in args.masses.split(',')] if args.masses else None
        extract(args.run_dir, args.test_dir, settings, masses=masses, start=args.start,
                frozen_dir=args.frozen_dir, pattern=args.step_pattern, n_steps=args.steps)
    else:
        replayed, _ = replay(load_recording(args.test_dir))
        deviation_table(load_golden(args.test_dir), replayed)


if __name__ == '__main__':
    sys.exit(main())
