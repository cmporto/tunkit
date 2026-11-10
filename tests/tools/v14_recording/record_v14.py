#!/usr/bin/env python3
"""
Developer script: record reference MWNEB runs with the v14 god class, for
tests/mwneb_runner_v14_replay.

fake_orca.py stands in for ORCA (model_potential.py, an ammonia-like double
well). Every calculation v14 makes is read back with the program's own
parsers (orca.read_input_xyz, orca.orca_engrad_parser) and saved, per case,
as <case>.json in the test folder, in the format runner.run_neb_from_json
reads. v14's final band, energies, fmax history and convergence go to
golden.json.

Usage (from anywhere):
    python record_v14.py /path/to/tunnel_code_v14.py [test_folder]

test_folder defaults to ../../mwneb_runner_v14_replay.
"""
import contextlib
import io
import json
import os
import shutil
import stat
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_potential as mp  # noqa: E402

from tunkit.io import orca  # noqa: E402
from tunkit.neb import neb_io  # noqa: E402
from tunkit.path.reaction_path import ReactionPath  # noqa: E402

ATOMIC_NUMBERS = [7, 1, 1, 1]

# What ee_inputs.make_header_orca(make_header_neb=True) writes for these options.
HEADER = """  ! engrad tightscf
!HF STO-3G

%maxcore 1000
%pal nprocs 1 end
* xyz 0 1
"""

COMMON = dict(n_images=9, k_spring=0.1, max_steps=30, max_move=0.2, save_every=10)

CASES = {
    'vpo_global':        dict(COMMON, optimizer='vpo', scope='global'),
    'vpo_local':         dict(COMMON, optimizer='vpo', scope='local'),
    'fire_local':        dict(COMMON, optimizer='fire', scope='local'),
    'fire_global_split': dict(COMMON, optimizer='fire', scope='global', split_forces=True),
    'lbfgs':             dict(COMMON, optimizer='lbfgs'),
    'vpo_split_precond': dict(COMMON, optimizer='vpo', scope='global',
                              split_forces=True, precondition=True),
    'vpo_align_every':   dict(COMMON, optimizer='vpo', scope='global', align_every_step=True),
    # warm start: a 6-frame rough band, aligned and resampled to 11 images
    'warm_resample':     dict(COMMON, optimizer='vpo', scope='global'),
}
WARM = {'warm_resample': dict(n_resample=11, align=True)}

# Also run through engine.run_orca_engrad and a generated driver in case.py
PLUMBING_CASE = 'vpo_global'


def write_warm_xyz(filename):
    """A rough 6-frame band: Cartesian interpolation plus a small fixed distortion."""
    R, P = mp.endpoints()
    rng = np.random.default_rng(42)
    sym = {7: 'N', 1: 'H'}
    with open(filename, 'w') as fo:
        for k in range(6):
            a = k / 5
            c = (1 - a) * R + a * P
            if 0 < k < 5:
                c = c + rng.normal(scale=0.05, size=c.shape)
            c_ang = c * 0.529177210544
            fo.write(f"4\nframe {k}\n")
            for z, xyz in zip(ATOMIC_NUMBERS, c_ang):
                fo.write(f" {sym[z]} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n")


def main(v14_file, test_dir):
    sys.path.insert(0, os.path.dirname(os.path.abspath(v14_file)))
    v14 = __import__(os.path.splitext(os.path.basename(v14_file))[0])

    work = os.path.join(HERE, '_v14_scratch')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    launcher = os.path.join(work, 'orca')
    with open(launcher, 'w') as fo:
        fo.write(f'#!/bin/sh\nexec "{sys.executable}" "{os.path.join(HERE, "fake_orca.py")}" "$@"\n')
    os.chmod(launcher, os.stat(launcher).st_mode | stat.S_IEXEC)
    warm_xyz = os.path.join(work, 'rough_band.xyz')
    write_warm_xyz(warm_xyz)

    os.makedirs(test_dir, exist_ok=True)
    for f in os.listdir(test_dir):
        if f.endswith('.json'):
            os.remove(os.path.join(test_dir, f))

    golden = {}
    for name, opts in CASES.items():
        wkb = v14.WKB_Tunneling_Analysis()
        wkb.atomic_numbers = np.array(ATOMIC_NUMBERS)
        wkb.get_masses()
        wkb.neb_header = HEADER

        # collect every calculation v14 makes, read back with the program's parsers
        calcs = {'image': [], 'step': [], 'coords': [], 'energy': [], 'gradient': []}
        n_steps = {'k': 0}
        original = wkb._neb_run_parallel

        def recording_run_parallel(orca_path, inp_files, n_parallel,
                                   _orig=original, _c=calcs, _n=n_steps):
            failed = _orig(orca_path, inp_files, n_parallel)
            frozen = os.path.basename(os.path.dirname(inp_files[0][1])) == 'frozen'
            for i, inp in sorted(inp_files):
                _, coords = orca.read_input_xyz(inp)
                gradient, energy = orca.orca_engrad_parser(inp[:-4] + '.engrad')
                _c['image'].append(int(i))
                _c['step'].append(-1 if frozen else _n['k'])
                _c['coords'].append(coords.tolist())
                _c['energy'].append(energy)
                _c['gradient'].append(np.array(gradient).reshape(coords.shape).tolist())
            if not frozen:
                _n['k'] += 1
            return failed

        wkb._neb_run_parallel = recording_run_parallel

        run_neb = dict(opts)
        if name in WARM:
            wkb.load_neb_path(warm_xyz, verbose=False, **WARM[name])
            reactant = product = None
            loaded = neb_io.load_neb_path(ReactionPath(), warm_xyz, verbose=False)
            run_neb.update(initial_band=loaded.raw_coord.tolist(),
                           n_images=WARM[name]['n_resample'],
                           align_initial_band=WARM[name]['align'])
        else:
            reactant, product = mp.endpoints()
            run_neb.update(reactant_coords=reactant.tolist(), product_coords=product.tolist())

        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            wkb.run_neb(reactant_coords=reactant, product_coords=product,
                        orca_path=launcher, n_parallel=2,
                        scratch_dir=os.path.join(work, name), verbose=True, **opts)

        record = {'description': f"v14 run '{name}' (fake ORCA, model potential)",
                  'atomic_numbers': ATOMIC_NUMBERS,
                  'masses': np.asarray(wkb.masses).tolist(),
                  'orca_header': HEADER,
                  'run_neb': run_neb,
                  'calculations': calcs}
        with open(os.path.join(test_dir, f'{name}.json'), 'w') as fo:
            json.dump(record, fo)

        golden[f'{name}_band'] = np.asarray(wkb.raw_coord).tolist()
        golden[f'{name}_energies'] = np.asarray(wkb.energies).tolist()
        golden[f'{name}_fmax_history'] = [float(x) for x in wkb.neb_fmax_history]
        golden[f'{name}_converged'] = bool(wkb.neb_converged)
        print(f"{name:20s} steps={len(wkb.neb_fmax_history):3d} converged={wkb.neb_converged} "
              f"calculations={len(calcs['image'])}")

    # case.py runs PLUMBING_CASE through the ORCA calculator and a driver too: same v14 answer
    for route in ('orca_engine', 'driver'):
        for key in ('band', 'energies', 'fmax_history', 'converged'):
            golden[f'{PLUMBING_CASE}_{route}_{key}'] = golden[f'{PLUMBING_CASE}_{key}']
    with open(os.path.join(test_dir, 'golden.json'), 'w') as fo:
        json.dump(golden, fo, indent=1, sort_keys=True)
    shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {len(CASES)} recordings and golden.json in {test_dir}")


if __name__ == '__main__':
    default_test = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'mwneb_runner_v14_replay')
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else default_test)
