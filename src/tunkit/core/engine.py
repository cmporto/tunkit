import os
import json
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tunkit.io import orca








def run_orca_parallel(orca_path, inp_files, n_parallel):
    """
    Run ORCA jobs in parallel using threads.
    
    Each ORCA process runs in the directory of its input file. Output 
    is captured to a .out file alongside the input.
    
    Parameters
    ----------
    orca_path : str
        Full path to the ORCA executable.
    inp_files : list of (int, str)
        Pairs of (image_index, input_file_path) from _neb_write_inputs
        or other file input generators.
    n_parallel : int
        Maximum number of concurrent ORCA jobs.
        
    Returns
    -------
    failed : list of int
        Indices of images whose ORCA jobs returned non-zero exit codes.
    """
    
    def _run_one(idx, inp_file):
        work_dir = os.path.dirname(inp_file)
        inp_name = os.path.basename(inp_file)
        out_name = inp_name.replace('.inp', '.out')
        out_path = os.path.join(work_dir, out_name)
        
        with open(out_path, 'w') as outf:
            result = subprocess.run(
                [orca_path, inp_name],
                cwd=work_dir,
                stdout=outf,
                stderr=subprocess.STDOUT,
            )
        return idx, result.returncode
    
    failed = []
    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = {
            executor.submit(_run_one, idx, path): idx
            for idx, path in inp_files
        }
        for future in as_completed(futures):
            idx, retcode = future.result()
            if retcode != 0:
                failed.append(idx)
                print(f"  WARNING: ORCA failed for image {idx} "
                        f"(exit code {retcode})")
    
    return failed








def run_orca_engrad(images_cart, indices, work_dir, header, atomic_numbers,
                    orca_path='orca', n_parallel=4):
    """
    ORCA calculator: write image_XXX.inp, run ORCA in parallel, read image_XXX.engrad.

    Parameters
    ----------
    header : str
        ORCA header ending in '* xyz charge mult', from
        ee_inputs.make_header_orca(make_header_neb=True).
    atomic_numbers : sequence of int
    orca_path : str
        Full path to ORCA (needed for %pal).
    n_parallel : int
        Concurrent ORCA jobs.
    """
    inp_files = orca.write_engrad_inputs(images_cart, work_dir, header,
                                         atomic_numbers, indices=indices)
    failed = run_orca_parallel(orca_path, inp_files, min(n_parallel, len(inp_files)))
    if failed:
        raise RuntimeError(
            f"ORCA failed for image(s) {sorted(failed)}. Check {work_dir}/")

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








def run_json_engrad(images_cart, indices, work_dir, json_file, tol=1e-7):
    """
    Calculator that reads energies and gradients from a recorded run
    instead of computing them.

    For each geometry asked for, the recorded calculation of the same image
    whose geometry is closest is used, if no coordinate differs by more than
    tol (bohr). Otherwise the run has left the recording, and a ValueError
    says which image and how far.

    Parameters
    ----------
    json_file : str
        A recorded run: its 'calculations' entry holds, for every calculation,
        'image' (int), 'step' (int, -1 for the endpoints), 'coords'
        (N_atoms x 3, bohr), 'energy' (Eh) and 'gradient' (N_atoms x 3, Eh/bohr).
        Written by the developer scripts that save a run.
    tol : float
        Largest coordinate difference (bohr) still treated as the same geometry.
    """
    rec = _load_json_calculations(json_file)
    energies = np.zeros(len(indices))
    gradients = np.zeros((len(indices),) + rec['coords'].shape[1:])
    for k, i in enumerate(indices):
        rows = np.flatnonzero(rec['image'] == i)
        if rows.size == 0:
            raise ValueError(f"No recorded calculation for image {i} in {json_file}.")
        dev = np.max(np.abs(rec['coords'][rows] - np.asarray(images_cart[i])), axis=(1, 2))
        best = int(np.argmin(dev))
        if dev[best] > tol:
            raise ValueError(
                f"Image {i}: the closest recorded geometry (step {rec['step'][rows[best]]}) "
                f"is {dev[best]:.2e} bohr away, more than tol = {tol:.0e}. "
                f"The run has left the recording here.")
        energies[k] = rec['energy'][rows[best]]
        gradients[k] = rec['gradient'][rows[best]]
    return energies, gradients








_JSON_CACHE = {}
def _load_json_calculations(json_file):
    """The 'calculations' of a recorded run as arrays, read once per file version."""
    key = (os.path.abspath(json_file), os.path.getmtime(json_file))
    if key not in _JSON_CACHE:
        with open(json_file) as fo:
            calcs = json.load(fo)['calculations']
        _JSON_CACHE[key] = {'image': np.asarray(calcs['image'], dtype=int),
                            'step': np.asarray(calcs['step'], dtype=int),
                            'coords': np.asarray(calcs['coords'], dtype=float),
                            'energy': np.asarray(calcs['energy'], dtype=float),
                            'gradient': np.asarray(calcs['gradient'], dtype=float)}
    return _JSON_CACHE[key]