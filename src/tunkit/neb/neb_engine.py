"""
Energy + gradient engines for the MWNEB runner.

An engine is any function with this signature:

    engine(images_cart, indices, work_dir) -> (energies, gradients)

    images_cart : list of ndarray (N_atoms, 3), the whole band in bohr
    indices     : list of int, the images to compute (others are ignored)
    work_dir    : str, a folder the engine may use for its files
    energies    : ndarray (len(indices),)             Eh
    gradients   : ndarray (len(indices), N_atoms, 3)  Eh/bohr  (dE/dx, not the force)

Row k of the outputs belongs to image indices[k]. The runner does the mass
weighting and the sign, so an engine only reports what the electronic
structure program prints. To use another program, write one function with
this signature and pass it as run_neb(..., engine=my_engine).

The options an engine needs besides those three arguments (header, paths,
parallel jobs...) are bound with functools.partial, e.g.

    functools.partial(engines.orca_engrad, header=h, atomic_numbers=z)
"""

import hashlib
import os

import numpy as np

from tunkit.core import engine as run_engine
from tunkit.io import orca
from tunkit.neb import neb_io








def orca_engrad(images_cart, indices, work_dir, header, atomic_numbers,
                orca_path='orca', n_parallel=4):
    """
    ORCA engine: write image_XXX.inp, run ORCA in parallel, read image_XXX.engrad.

    Parameters
    ----------
    header : str
        ORCA header ending in '* xyz charge mult', from
        ee_inputs.make_header_orca(make_header_neb=True).
    atomic_numbers : sequence of int
    orca_path : str
        Full path to ORCA (needed for %pal), or to tests/tools/orca_record.py
        to record the run.
    n_parallel : int
        Concurrent ORCA jobs.
    """
    # write inputs
    inp_files = neb_io._neb_write_inputs(
        images_cart, work_dir, header, atomic_numbers, indices=indices)

    # then run
    failed = run_engine.run_orca_parallel(
        orca_path, inp_files, min(n_parallel, len(inp_files)))
    if failed:
        raise RuntimeError(
            f"ORCA failed for image(s) {sorted(failed)}. Check {work_dir}/")
    
    # grab the results in the engrad files (energies and gradients)
    n_atoms = len(atomic_numbers)
    energies = np.zeros(len(indices))
    gradients = np.zeros((len(indices), n_atoms, 3))
    for k, i in enumerate(indices):
        engrad_path = os.path.join(work_dir, f"image_{i:03d}.engrad")
        if not os.path.exists(engrad_path):
            raise FileNotFoundError(
                f"Missing .engrad for image {i}. "
                f"Check {work_dir}/image_{i:03d}.out")
        gradient, energy = orca.orca_engrad_parser(engrad_path)
        energies[k] = energy
        gradients[k] = np.array(gradient).reshape(n_atoms, 3)
    return energies, gradients








def input_hash(text):
    """SHA-256 of an input file's text, line endings normalized (same as orca_record.py)."""
    return hashlib.sha256(text.replace('\r\n', '\n').encode()).hexdigest()








def replay_engrad(images_cart, indices, work_dir, header, atomic_numbers,
                  archive_dir):
    """
    Replay engine: look each image's result up in a recorded archive instead
    of running ORCA.

    The archive is written by tests/tools/orca_record.py: <hash>.engrad, where
    <hash> is input_hash() of the input text. This engine builds the same text
    with neb_io._neb_input_text, so a run that reproduces the recorded one
    finds every image. The first image it cannot find is where the two runs
    diverged, and the error says which.
    """
    n_atoms = len(atomic_numbers)
    energies = np.zeros(len(indices))
    gradients = np.zeros((len(indices), n_atoms, 3))
    for k, i in enumerate(indices):
        text = neb_io._neb_input_text(images_cart[i], header, atomic_numbers)
        h = input_hash(text)
        engrad_path = os.path.join(archive_dir, h + '.engrad')
        if not os.path.exists(engrad_path):
            raise KeyError(
                f"No recorded result for image {i} (input hash {h[:12]}...). "
                f"This run has diverged from the recording here.")
        gradient, energy = orca.orca_engrad_parser(engrad_path)
        energies[k] = energy
        gradients[k] = np.array(gradient).reshape(n_atoms, 3)
    return energies, gradients








def recorded_steps_engrad(images_cart, indices, work_dir, recording, calls):
    """
    Replay engine for a run saved step by step (frozen/ + current_NNN/ folders).

    Unlike replay_engrad, results are looked up by call order, not by input
    text: the first call returns the endpoints, call k+1 returns step k. The
    recorded result is returned whatever geometry the runner asks for, so a
    replay that drifts keeps going and the drift can be measured afterwards.

    Parameters
    ----------
    recording : dict
        'endpoints': {'indices': [0, N-1], 'energies': (2,), 'gradients': (2, N_atoms, 3)}
        'steps':     list of the same, one per step, for the movable images.
        Built by tests/tools/neb_trajectory.py from the saved folders.
    calls : list
        Each call appends {'indices': [...], 'coords': (n, N_atoms, 3)}: the
        geometries the runner asked for. Compare them with the recorded
        geometries to see how far the replay is from the original run.
    """
    indices = list(indices)
    if not calls:
        block = recording['endpoints']
        label = 'the endpoints'
    else:
        k = len(calls) - 1
        if k >= len(recording['steps']):
            raise IndexError(
                f"The run asked for step {k}, but only "
                f"{len(recording['steps'])} steps were recorded.")
        block = recording['steps'][k]
        label = f'step {k}'
    if list(block['indices']) != indices:
        raise ValueError(
            f"The run asked for images {indices} at {label}, but the recording "
            f"has images {list(block['indices'])}.")
    calls.append({'indices': indices,
                  'coords': np.array([images_cart[i] for i in indices])})
    return np.asarray(block['energies'], dtype=float), np.asarray(block['gradients'], dtype=float)
