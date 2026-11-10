import numpy as np
import os
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from tunkit.core import constants as const
from tunkit.core import elements as elem
from tunkit.core import geometry as geom
from tunkit.io import orca
from tunkit.neb import band








def _neb_parse_results(scratch_dir, n_images, masses, indices=None):
    """
    Parse ORCA engrad files for NEB images.
    
    Reads energies and gradients, converts Cartesian forces to 
    mass-weighted space: F_mw = (-gradient) / sqrt(m).
    
    Parameters
    ----------
    scratch_dir : str
        Directory containing .engrad files from engine.run_orca_parallel.
    n_images : int
        Total number of images (including endpoints).
    indices : list of int, optional
        Which image indices to parse. If None, parses all interior 
        images (1 to N-2).
        
    Returns
    -------
    energies : ndarray, shape (N_images,)
        Energy of each image in Hartree. Only entries in indices 
        are filled; the rest are zero.
    forces_mw : list of ndarray, each (N_atoms, 3)
        True force in mass-weighted space for each image. Only 
        entries in indices are filled; the rest are zeros.
        
    Raises
    ------
    FileNotFoundError
        If a requested image's .engrad file is missing.
    """
    n_atoms = len(masses)
    sqrt_m = np.sqrt(masses)[:, np.newaxis]
    
    energies = np.zeros(n_images)
    forces_mw = [np.zeros((n_atoms, 3)) for _ in range(n_images)]
    
    if indices is None:
        indices = list(range(1, n_images - 1))
    
    for i in indices:
        engrad_path = os.path.join(
            scratch_dir, f"image_{i:03d}.engrad"
        )
        
        if not os.path.exists(engrad_path):
            raise FileNotFoundError(
                f"Missing .engrad for image {i}. "
                f"Check {scratch_dir}/image_{i:03d}.out"
            )
        
        gradient, energy = orca.orca_engrad_parser(engrad_path)
        energies[i] = energy
        
        f_cart = -np.array(gradient).reshape(n_atoms, 3)
        forces_mw[i] = f_cart / sqrt_m
    
    return energies, forces_mw








# def _neb_run_parallel(orca_path, inp_files, n_parallel):
#     """
#     Run ORCA jobs in parallel using threads.
    
#     Each ORCA process runs in the directory of its input file. Output 
#     is captured to a .out file alongside the input.
    
#     Parameters
#     ----------
#     orca_path : str
#         Full path to the ORCA executable.
#     inp_files : list of (int, str)
#         Pairs of (image_index, input_file_path) from _neb_write_inputs.
#     n_parallel : int
#         Maximum number of concurrent ORCA jobs.
        
#     Returns
#     -------
#     failed : list of int
#         Indices of images whose ORCA jobs returned non-zero exit codes.
#     """
    
#     def _run_one(idx, inp_file):
#         work_dir = os.path.dirname(inp_file)
#         inp_name = os.path.basename(inp_file)
#         out_name = inp_name.replace('.inp', '.out')
#         out_path = os.path.join(work_dir, out_name)
        
#         with open(out_path, 'w') as outf:
#             result = subprocess.run(
#                 [orca_path, inp_name],
#                 cwd=work_dir,
#                 stdout=outf,
#                 stderr=subprocess.STDOUT,
#             )
#         return idx, result.returncode
    
#     failed = []
#     with ThreadPoolExecutor(max_workers=n_parallel) as executor:
#         futures = {
#             executor.submit(_run_one, idx, path): idx 
#             for idx, path in inp_files
#         }
#         for future in as_completed(futures):
#             idx, retcode = future.result()
#             if retcode != 0:
#                 failed.append(idx)
#                 print(f"  WARNING: ORCA failed for image {idx} "
#                         f"(exit code {retcode})")
    
#     return failed








# def _neb_write_inputs(images_cart, 
#                       scratch_dir, 
#                       neb_header, 
#                       atomic_numbers,
#                       indices=None):
#     """
#     Write ORCA engrad input files for NEB images.
    
#     Uses self.neb_header from make_header_orca(). Coordinates are written 
#     in Angstrom (ORCA's default for xyz input).
    
#     Parameters
#     ----------
#     images_cart : list of ndarray, each (N_atoms, 3)
#         Image coordinates in bohr.
#     scratch_dir : str
#         Directory for input/output files. Created if it doesn't exist.
#     indices : list of int, optional
#         Which image indices to write. If None, writes all interior 
#         images (1 to N-2). Use [0, N-1] for endpoints. (Future implementation)
        
#     Returns
#     -------
#     inp_files : list of (int, str)
#         Pairs of (image_index, input_file_path).
#     """
#     os.makedirs(scratch_dir, exist_ok=True)
    
#     if indices is None:
#         indices = list(range(1, len(images_cart) - 1))
    
#     inp_files = []
#     # for i in indices:
#     #     coords_ang = images_cart[i] * const.bohr_to_ang
#     #     inp_path = os.path.join(scratch_dir, f"image_{i:03d}.inp")
        
#     #     with open(inp_path, 'w') as f:
#     #         f.write(neb_header)
#     #         for n, atom in enumerate(coords_ang):
#     #             sym = elem.elements[atomic_numbers[n]][0]
#     #             f.write(f' {sym:2s} {atom[0]:16.10f} {atom[1]:16.10f} {atom[2]:16.10f}\n')
#     #         f.write('*\n')
        
#     #     inp_files.append((i, inp_path))

#     for i in indices:
#         inp_path = os.path.join(scratch_dir, f"image_{i:03d}.inp")
        
#         with open(inp_path, 'w') as f:
#             f.write(_neb_input_text(images_cart[i], neb_header, atomic_numbers))
        
#         inp_files.append((i, inp_path))

#     return inp_files








# def _neb_input_text(coords_cart, neb_header, atomic_numbers):
#     """
#     Text of one ORCA engrad input (header + xyz block in Angstrom).

#     Kept separate from _neb_write_inputs so the replay engine
#     (engines.replay_engrad) builds exactly the same text, and therefore
#     the same hash, as the file ORCA received.
#     """
#     coords_ang = coords_cart * const.bohr_to_ang
#     lines = [neb_header]
#     for n, atom in enumerate(coords_ang):
#         sym = elem.elements[atomic_numbers[n]][0]
#         lines.append(f' {sym:2s} {atom[0]:16.10f} {atom[1]:16.10f} {atom[2]:16.10f}\n')
#     lines.append('*\n')
#     return ''.join(lines)









def load_neb_path(path,
                  source, 
                  n_resample=None, 
                  align=False, 
                  custom_masses=None, 
                  verbose=True):
    """
    Load an existing reaction path as starting point for run_neb().
    
    Reads endpoint structures and intermediate images from various 
    formats, aligns them, converts to mass-weighted coordinates, and 
    optionally resamples to a different number of images. The result 
    is stored in self.neb_images_mw, which run_neb() picks up as a 
    warm start.
    
    This is the recommended workflow when you already have a Cartesian 
    NEB or IRC path and want to re-optimize it in mass-weighted space:
    
    Example:

        wkb.load_neb_path('previous_neb.xyz', n_resample=20)
        wkb.run_neb()
    
    Parameters
    ----------
    source : str
        Path to the input. Accepted formats:
        
        - Multi-image .xyz file: standard XYZ trajectory with energies 
            on the comment line. Same format as extract_coord_energy_xyz().
            
        - Directory of .xyz files: individual images as 000.xyz, 
            001.xyz, etc. Sorted alphabetically.
            
        - Directory of ORCA .hess/.engrad files: the projection folder 
            format from make_projection_files_orca(). Images read from 
            .hess files (which contain geometry).
            
        - restart.json: from a previous run_neb() (in scratch_dir/).
        
    n_resample : int, optional
        Resample the path to this many images using cubic spline 
        interpolation along the mass-weighted arc length. If None, 
        uses the images as read.
        
        Resampling produces equal spacing in mass-weighted arc length, 
        which means denser coverage near light atoms with large 
        displacements (the tunneling hydrogen).
        
    align : bool
        Sequentially align images via Kabsch before converting to 
        mass-weighted coordinates. Default True. Disable only if the 
        path is already consistently aligned.
        
    masses : list or ndarray, optional
        Override atomic masses for isotopic substitution. If None, 
        uses standard masses from the element table.
        
    verbose : bool
        Print summary. Default True.
        
    Attributes Set
    --------------
    self.neb_images_mw : list of ndarray, each (N_atoms, 3)
        Mass-weighted coordinates for each image. Detected by 
        run_neb() as a warm start.
    self.atomic_numbers : ndarray
        Populated if not already set.
    self.masses : ndarray
        Populated if not already set, or overridden if masses given.
    """
    if verbose:
        print(f"Loading NEB path from: {source}")

    # Detect format and read
    if os.path.isfile(source):
        if source.endswith('.xyz'):
            all_coords_bohr, energies, masses, atomic_numbers = _neb_read_multi_xyz(source, verbose)
        else:
            # Try as xyz (there may be a different name but same structure)
            # Could give rise to weird issues if file is not multimage xyz...
            all_coords_bohr, energies, masses, atomic_numbers = _neb_read_multi_xyz(source, verbose)
    elif os.path.isdir(source):
        # I could make use that we may already have the hess and engrad files and useit as a
        # warm start foir the neb runner... maybe [TODO]
        all_coords_bohr, energies, masses, atomic_numbers = _neb_read_directory(source, verbose)
    else:
        raise FileNotFoundError(f"Source not found: {source}")
    
    n_frames = len(all_coords_bohr)
    
    # Override masses if requested (isotopic substitution)
    if custom_masses is not None:
        masses = np.array(custom_masses)
        if verbose:
            print(f"  Using custom masses (isotopic substitution).")

    # There is no scenario where the masses array wouldn't be populated.
    # [TODO] maybe could be used to validate the array given by the user...
    # if len(path.masses) == 0:
    #     raise ValueError(
    #         "Masses not set. The source did not contain mass info "
    #         "and self.masses is empty. Call elements.get_masses() first or "
    #         "pass masses= explicitly."
        # )
    
    # Sequential alignment
    if align:
        aligned = [all_coords_bohr[0]]
        for i in range(1, n_frames):
            img_aligned, _, _ = geom.kabsch_alignment_cm(
                all_coords_bohr[i], aligned[i - 1], masses
            )
            aligned.append(img_aligned)
        all_coords_bohr = aligned
        if verbose:
            print(f"  Aligned {n_frames} images (sequential Kabsch).")
    
    # Resample
    if n_resample is not None and n_resample != n_frames:
        all_coords_bohr = band._neb_resample_path(
            all_coords_bohr, n_resample, masses
        )
        n_frames = n_resample
        energies=None
        if verbose:
            print(f"  Resampled to {n_frames} images. "
                  "(energies dropped as they belong to the original images).")
    
    # Convert to mass-weighted
    sqrt_m = np.sqrt(masses)[:, np.newaxis]
    neb_images_mw = [coords * sqrt_m for coords in all_coords_bohr]
    
    # printing verbose information
    if verbose:
        ds_total = sum(
            np.linalg.norm(
                neb_images_mw[i+1] - neb_images_mw[i]
            )
            for i in range(n_frames - 1)
        )
        print(f"  Total MW path length: {ds_total:.4f} "
                f"bohr*amu^(1/2)")
        print(f"  {n_frames} images ready. ")

    all_coords_bohr = np.array(all_coords_bohr)

    path = replace(path,
                   atomic_numbers=atomic_numbers, 
                   raw_coord_ang=all_coords_bohr * const.bohr_to_ang, 
                   raw_coord=all_coords_bohr, 
                   mass_scaled_coord=neb_images_mw,
                   energies=energies, 
                   masses=masses)

    return path
    # return neb_images_mw








def _neb_read_multi_xyz(filename, verbose=True):
    """
    Read a multi-frame XYZ file into a list of coordinate arrays.
    
    Also populates self.atomic_numbers and self.masses if they are 
    not already set. Energies on the comment line are read into 
    self.energies if available.
    
    Parameters
    ----------
    filename : str
    verbose : bool
    
    Returns
    -------
    all_coords_bohr : list of ndarray, each (N_atoms, 3)
    """
    all_coords = []
    energies = []
    first_frame_an = []
    
    with open(filename) as fo:
        lines = fo.readlines()
    
    i = 0
    frame = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        try:
            n_atoms = int(line)
        except ValueError:
            i += 1
            continue
        
        # Comment line so trying to extract energy
        i += 1
        if i < len(lines):
            comment = lines[i].strip()
            energy = _neb_parse_comment_energy(comment)
            energies.append(energy)
        
        # Atom lines
        frame_an = []
        frame_coords = []
        for _ in range(n_atoms):
            i += 1
            if i >= len(lines):
                break
            parts = lines[i].split()
            sym = parts[0]
            an = elem.search_element(sym)
            if an is None:
                # Could be a numeric atomic number
                try:
                    an = int(sym)
                except ValueError:
                    raise ValueError(
                        f"Unknown atom '{sym}' in frame {frame}"
                    )
            frame_an.append(an)
            frame_coords.append([
                float(parts[1]), float(parts[2]), float(parts[3])
            ])
        
        all_coords.append(
            np.array(frame_coords) * const.ang_to_bohr
        )
        
        if frame == 0:
            first_frame_an = frame_an
        
        frame += 1
        i += 1
    
    # Populate class attributes if needed
    # if len(atomic_numbers) == 0:
    atomic_numbers = np.array(first_frame_an)
    # if len(self.masses) == 0:
    masses = elem.get_masses(atomic_numbers)
    
    # Store energies if parsed
    energies = _neb_check_energies(energies, list(range(len(energies))), filename, verbose)
    # valid_energies = [e for e in energies if e is not None] # packed into _neb_check_energies
    # if len(valid_energies) == len(all_coords):
    #     energies = np.array(energies)
    # else:
    #     energies = None
    
    if verbose:
        print(f"  Read {frame} frames from {filename}")
    
    return all_coords, energies, masses, atomic_numbers








def _neb_read_directory(dirpath, verbose=True):
    """
    Read a directory of NEB images.
    
    Tries two strategies:
    
    1. ORCA .hess files (from make_projection_files_orca output):
        reads geometry from the $atoms block. Paired .engrad files 
        provide energies.
        
    2. Individual .xyz files: sorted alphabetically.
    
    Parameters
    ----------
    dirpath : str
    verbose : bool
    
    Returns
    -------
    all_coords_bohr : list of ndarray, each (N_atoms, 3)
    """
    all_files = sorted(os.listdir(dirpath))
    
    # Strategy 1: .hess files (ORCA projection folder)
    hess_files = [f for f in all_files if f.endswith('.hess')]
    if hess_files:
        if verbose:
            print(f"  Found {len(hess_files)} .hess files in {dirpath}")
        
        all_coords = []
        energies = []
        
        for hf in sorted(hess_files):
            hess_path = os.path.join(dirpath, hf)
            masses, atoms, coords, _, _, _ = orca.read_orca_data(hess_path) # returns in bohrs
            all_coords.append(coords)  # already in bohr **
            
            # Try to get energy from .engrad files
            stem = os.path.splitext(hf)[0]
            engrad_path = os.path.join(dirpath, stem + '.engrad')
            if os.path.exists(engrad_path):
                gradient, energy = orca.orca_engrad_parser(engrad_path)
                energies.append(energy)
            else:
                energies.append(None)
        
        # Populate class attributes with the information from files
        # if len(self.atomic_numbers) == 0:
            atomic_numbers = np.array([
                elem.search_element(a) for a in atoms
            ])
        # if len(self.masses) == 0:
            masses = np.array(masses)

        energies = _neb_check_energies(energies, sorted(hess_files), dirpath, verbose)
        # valid_e = [e for e in energies if e is not None] # packed into _neb_check_energies
        # if len(valid_e) == len(all_coords):
        #     energies = np.array(energies)
        # else:
        #     energies = None
        
        return all_coords, energies, masses, atomic_numbers
    
    # Strategy 2: .xyz files
    xyz_files = [f for f in all_files if f.endswith('.xyz')]
    if xyz_files:
        if verbose:
            print(f"  Found {len(xyz_files)} .xyz files in {dirpath}")

        all_coords = []
        energies = []
        for xf in sorted(xyz_files):
            # Read each xyz using the multi-image reader 
            # but it works for single files too
            xyz_path = os.path.join(dirpath, xf)
            frame_coords, frame_e, masses, atomic_numbers = _neb_read_multi_xyz(
                xyz_path, verbose=False
            )
            # Take the first (and usually only) image
            if frame_coords:
                all_coords.append(frame_coords[0])
                energies.append(frame_e[0] if frame_e is not None else None)

        energies = _neb_check_energies(energies, sorted(xyz_files), dirpath, verbose)
        # if any(e is None for e in energies):
        #     energies = None
        # else:
        #     energies = np.array(energies)

        return all_coords, energies, masses, atomic_numbers

    raise FileNotFoundError(
        f"No .hess or .xyz files found in {dirpath}"
    )








# Method is kinda flimsy but works well with our current formats, so...
def _neb_parse_comment_energy(comment):
    """
    Try to extract a Hartree energy from an XYZ comment line.
    
    Handles the following formats:
        'Energy = -123.456'          (our format + common convention)
        'blah blah blah blah blah -123.456'   (field index 5, ORCA output)
    
    Returns None if no energy found.
    """
    if not comment:
        return None
    
    parts = comment.split()
    
    # Pattern: 'Energy = VALUE' or 'E = VALUE'
    for k, token in enumerate(parts):
        if token.lower() in ('energy', 'e') and k + 1 < len(parts):
            next_tok = parts[k + 1]
            if next_tok == '=' and k + 2 < len(parts):
                try:
                    return float(parts[k + 2])
                except ValueError:
                    pass
            else:
                try:
                    return float(next_tok)
                except ValueError:
                    pass
    
    # Orca trj format for now energy at index 5
    if len(parts) > 5:
        try:
            val = float(parts[5])
            if -10000 < val < 0: # kinda arbitrary, but well
                return val
        except ValueError:
            pass
    
    # Fallback: any negative float in Hartree range ¯\_(՞ ◡ ՞)_/¯  (maybe warn?)
    for token in parts:
        try:
            val = float(token)
            if -10000 < val < -1:
                return val
        except ValueError:
            continue
    
    return None







# [TODO] also save the optimizer history to maybe use it...
def _neb_save_state(scratch_dir, 
                    images_mw, 
                    energies, 
                    fmax_history, 
                    step,
                    masses, 
                    atomic_numbers,
                    fmax_per_image=None,
                    job_prefix=None):
    """
    Save NEB state for restart and a band trajectory for visualization.
    
    Writes:
        - scratch_dir/restart.json  (full state for resuming)
        - scratch_dir/band.xyz      (multi-frame XYZ for visualization)
    
    Parameters
    ----------
    scratch_dir : str
    images_mw : list of ndarray
    energies : ndarray
    fmax_history : list of float
    step : int
    """
    
    sqrt_m = np.sqrt(masses)[:, np.newaxis]
    
    # JSON restart
    restart = { # add the rest of the variables. It should carry everything!
        'step': step,
        'images_mw': [img.tolist() for img in images_mw],
        'energies': energies.tolist(),
        'fmax_history': fmax_history,
    }
    restart_path = os.path.join(scratch_dir, 'restart.json')
    with open(restart_path, 'w') as fo:
        json.dump(restart, fo)
    
    if fmax_per_image is None:
        fmax_per_image = {}
    
    if job_prefix:
        band_path = f"{job_prefix}_band.xyz"
        all_band_path = f"{job_prefix}_all_bands.xyz"
    else:
        band_path = os.path.join(scratch_dir, 'band.xyz')
        all_band_path = os.path.join(scratch_dir, 'all_bands.xyz')

    # Multistructure XYZ band
    # band_path = os.path.join(scratch_dir, 'band.xyz')
    # all_band_path = os.path.join(scratch_dir, 'all_bands.xyz')
    with open(band_path, 'w') as fo:
        for i, mw in enumerate(images_mw):
            cart_ang = (mw / sqrt_m) * const.bohr_to_ang
            n_at = len(atomic_numbers)
            fo.write(f"{n_at}\n")
            fo.write(f"Step {step} Image {i} "
                    f"Energy = {energies[i]:.10f}\n")
            for n in range(n_at):
                sym = elem.elements[atomic_numbers[n]][0]
                fo.write(f" {sym:2s} {cart_ang[n,0]:16.10f} "
                        f"{cart_ang[n,1]:16.10f} "
                        f"{cart_ang[n,2]:16.10f}\n")
    
    with open(all_band_path, 'a') as fo:
        for i, mw in enumerate(images_mw):
            cart_ang = (mw / sqrt_m) * const.bohr_to_ang
            n_at = len(atomic_numbers)
            fo.write(f"{n_at}\n")
            fo.write(f"Step {step} Image {i} "
                    f"Energy = {energies[i]:.10f}\n")
            for n in range(n_at):
                sym = elem.elements[atomic_numbers[n]][0]
                fo.write(f" {sym:2s} {cart_ang[n,0]:16.10f} "
                        f"{cart_ang[n,1]:16.10f} "
                        f"{cart_ang[n,2]:16.10f}\n")








def _neb_read_restart(filepath, masses, verbose=True):
    """
    Read a restart.json from a previous run_neb().
    
    Restores images in mass-weighted coordinates, converts back to 
    Cartesian so the standard load_neb_path pipeline (alignment, 
    resampling) can proceed.
    
    Parameters
    ----------
    filepath : str
    verbose : bool
    
    Returns
    -------
    all_coords_bohr : list of ndarray, each (N_atoms, 3)
    """
    
    with open(filepath) as f:
        data = json.load(f)
    
    images_mw = [np.array(img) for img in data['images_mw']]
    
    # We need masses to convert back to Cartesian
    # If masses aren't set yet, we can't proceed
    if len(masses) == 0:
        raise ValueError(
            "Cannot read restart without masses. Set self.masses "
            "or self.atomic_numbers + elements.get_masses() before loading."
        )
    
    sqrt_m = np.sqrt(masses)[:, np.newaxis]
    all_coords_bohr = [mw / sqrt_m for mw in images_mw]
    
    # Restore energies and history if present
    if 'energies' in data:
        energies = np.array(data['energies'])
    
    step = data.get('step', '?')
    if verbose:
        print(f"  Read restart from step {step} "
                f"({len(images_mw)} images)")
    
    return all_coords_bohr, energies








def _neb_check_energies(energies, labels, source, verbose=True):
    """
    Checking energies for a path or xyz file: it returns the energies
    as an numpy array if every energy has been succesfully loaded or
    None if it hasn't. It prints a warning if a part of the images
    doesn't have an energy, which usually means a parsing problem. 
    If there are no energies, it just returns None.
    """
    missing = [lab for lab, e in zip(labels, energies) if e is None]
    if not missing:
        return np.array(energies)
    if len(missing) < len(energies):
        print(f"  WARNING: no energy for {missing} in {source}. "
              f"Energies discarded for the whole path.")
    elif verbose:
        print(f"  No energies found in {source}.")
    return None








def _neb_read_input_xyz(inp_path):
    """
    Read the xyz block of an ORCA input (the inverse of _neb_input_text).

    Returns
    -------
    atomic_numbers : ndarray of int
    coords_bohr : ndarray (N_atoms, 3)
        The geometry ORCA received, converted from Angstrom to bohr.
    """
    atomic_numbers, coords, inside = [], [], False
    with open(inp_path) as fo:
        for line in fo:
            s = line.strip()
            if not inside:
                if s.replace(' ', '').lower().startswith('*xyz'):
                    inside = True
                continue
            if s.startswith('*'):
                break
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            an = elem.search_element(parts[0])
            if an is None:
                raise ValueError(f"Unknown element '{parts[0]}' in {inp_path}")
            atomic_numbers.append(an)
            coords.append([float(x) for x in parts[1:4]])
    if not coords:
        raise ValueError(f"No '* xyz' block found in {inp_path}")
    return np.array(atomic_numbers), np.array(coords) * const.ang_to_bohr








# class IoNebMixin:

    # def _neb_parse_results(self, scratch_dir, n_images, indices=None):
    #     """
    #     Parse ORCA engrad files for NEB images.
        
    #     Reads energies and gradients, converts Cartesian forces to 
    #     mass-weighted space: F_mw = (-gradient) / sqrt(m).
        
    #     Parameters
    #     ----------
    #     scratch_dir : str
    #         Directory containing .engrad files from _neb_run_parallel.
    #     n_images : int
    #         Total number of images (including endpoints).
    #     indices : list of int, optional
    #         Which image indices to parse. If None, parses all interior 
    #         images (1 to N-2).
            
    #     Returns
    #     -------
    #     energies : ndarray, shape (N_images,)
    #         Energy of each image in Hartree. Only entries in indices 
    #         are filled; the rest are zero.
    #     forces_mw : list of ndarray, each (N_atoms, 3)
    #         True force in mass-weighted space for each image. Only 
    #         entries in indices are filled; the rest are zeros.
            
    #     Raises
    #     ------
    #     FileNotFoundError
    #         If a requested image's .engrad file is missing.
    #     """
    #     n_atoms = len(self.masses)
    #     sqrt_m = np.sqrt(self.masses)[:, np.newaxis]
        
    #     energies = np.zeros(n_images)
    #     forces_mw = [np.zeros((n_atoms, 3)) for _ in range(n_images)]
        
    #     if indices is None:
    #         indices = list(range(1, n_images - 1))
        
    #     for i in indices:
    #         engrad_path = os.path.join(
    #             scratch_dir, f"image_{i:03d}.engrad"
    #         )
            
    #         if not os.path.exists(engrad_path):
    #             raise FileNotFoundError(
    #                 f"Missing .engrad for image {i}. "
    #                 f"Check {scratch_dir}/image_{i:03d}.out"
    #             )
            
    #         gradient, energy = self.orca_engrad_parser(engrad_path)
    #         energies[i] = energy
            
    #         f_cart = -np.array(gradient).reshape(n_atoms, 3)
    #         forces_mw[i] = f_cart / sqrt_m
        
    #     return energies, forces_mw








    # def _neb_run_parallel(self, orca_path, inp_files, n_parallel):
    #     """
    #     Run ORCA jobs in parallel using threads.
        
    #     Each ORCA process runs in the directory of its input file. Output 
    #     is captured to a .out file alongside the input.
        
    #     Parameters
    #     ----------
    #     orca_path : str
    #         Full path to the ORCA executable.
    #     inp_files : list of (int, str)
    #         Pairs of (image_index, input_file_path) from _neb_write_inputs.
    #     n_parallel : int
    #         Maximum number of concurrent ORCA jobs.
            
    #     Returns
    #     -------
    #     failed : list of int
    #         Indices of images whose ORCA jobs returned non-zero exit codes.
    #     """
        
    #     def _run_one(idx, inp_file):
    #         work_dir = os.path.dirname(inp_file)
    #         inp_name = os.path.basename(inp_file)
    #         out_name = inp_name.replace('.inp', '.out')
    #         out_path = os.path.join(work_dir, out_name)
            
    #         with open(out_path, 'w') as outf:
    #             result = subprocess.run(
    #                 [orca_path, inp_name],
    #                 cwd=work_dir,
    #                 stdout=outf,
    #                 stderr=subprocess.STDOUT,
    #             )
    #         return idx, result.returncode
        
    #     failed = []
    #     with ThreadPoolExecutor(max_workers=n_parallel) as executor:
    #         futures = {
    #             executor.submit(_run_one, idx, path): idx 
    #             for idx, path in inp_files
    #         }
    #         for future in as_completed(futures):
    #             idx, retcode = future.result()
    #             if retcode != 0:
    #                 failed.append(idx)
    #                 print(f"  WARNING: ORCA failed for image {idx} "
    #                       f"(exit code {retcode})")
        
    #     return failed








    # def _neb_write_inputs(self, images_cart, scratch_dir, indices=None):
    #     """
    #     Write ORCA engrad input files for NEB images.
        
    #     Uses self.neb_header from make_header_orca(). Coordinates are written 
    #     in Angstrom (ORCA's default for xyz input).
        
    #     Parameters
    #     ----------
    #     images_cart : list of ndarray, each (N_atoms, 3)
    #         Image coordinates in bohr.
    #     scratch_dir : str
    #         Directory for input/output files. Created if it doesn't exist.
    #     indices : list of int, optional
    #         Which image indices to write. If None, writes all interior 
    #         images (1 to N-2). Use [0, N-1] for endpoints. (Future implementation)
            
    #     Returns
    #     -------
    #     inp_files : list of (int, str)
    #         Pairs of (image_index, input_file_path).
    #     """
    #     os.makedirs(scratch_dir, exist_ok=True)
        
    #     if indices is None:
    #         indices = list(range(1, len(images_cart) - 1))
        
    #     inp_files = []
    #     for i in indices:
    #         coords_ang = images_cart[i] * const.bohr_to_ang
    #         inp_path = os.path.join(scratch_dir, f"image_{i:03d}.inp")
            
    #         with open(inp_path, 'w') as f:
    #             f.write(self.neb_header)
    #             for n, atom in enumerate(coords_ang):
    #                 sym = elem.elements[self.atomic_numbers[n]][0]
    #                 f.write(f' {sym:2s} {atom[0]:16.10f} {atom[1]:16.10f} {atom[2]:16.10f}\n')
    #             f.write('*\n')
            
    #         inp_files.append((i, inp_path))
        
    #     return inp_files








    # def _neb_save_state(self, scratch_dir, images_mw, energies, 
    #                     fmax_history, step, fmax_per_image=None,
    #                     job_prefix=None):
    #     """
    #     Save NEB state for restart and a band trajectory for visualization.
        
    #     Writes:
    #       - scratch_dir/restart.json  (full state for resuming)
    #       - scratch_dir/band.xyz      (multi-frame XYZ for visualization)
        
    #     Parameters
    #     ----------
    #     scratch_dir : str
    #     images_mw : list of ndarray
    #     energies : ndarray
    #     fmax_history : list of float
    #     step : int
    #     """
        
    #     sqrt_m = np.sqrt(self.masses)[:, np.newaxis]
        
    #     # JSON restart
    #     restart = {
    #         'step': step,
    #         'images_mw': [img.tolist() for img in images_mw],
    #         'energies': energies.tolist(),
    #         'fmax_history': fmax_history,
    #     }
    #     restart_path = os.path.join(scratch_dir, 'restart.json')
    #     with open(restart_path, 'w') as fo:
    #         json.dump(restart, fo)
        
    #     if fmax_per_image is None:
    #         fmax_per_image = {}
        
    #     if job_prefix:
    #         band_path = f"{job_prefix}_band.xyz"
    #         all_band_path = f"{job_prefix}_all_bands.xyz"
    #     else:
    #         band_path = os.path.join(scratch_dir, 'band.xyz')
    #         all_band_path = os.path.join(scratch_dir, 'all_bands.xyz')

    #     # Multistructure XYZ band
    #     # band_path = os.path.join(scratch_dir, 'band.xyz')
    #     # all_band_path = os.path.join(scratch_dir, 'all_bands.xyz')
    #     with open(band_path, 'w') as fo:
    #         for i, mw in enumerate(images_mw):
    #             cart_ang = (mw / sqrt_m) * const.bohr_to_ang
    #             n_at = len(self.atomic_numbers)
    #             fo.write(f"{n_at}\n")
    #             fo.write(f"Step {step} Image {i} "
    #                     f"Energy = {energies[i]:.10f}\n")
    #             for n in range(n_at):
    #                 sym = elem.elements[self.atomic_numbers[n]][0]
    #                 fo.write(f" {sym:2s} {cart_ang[n,0]:16.10f} "
    #                         f"{cart_ang[n,1]:16.10f} "
    #                         f"{cart_ang[n,2]:16.10f}\n")
        
    #     with open(all_band_path, 'a') as fo:
    #         for i, mw in enumerate(images_mw):
    #             cart_ang = (mw / sqrt_m) * const.bohr_to_ang
    #             n_at = len(self.atomic_numbers)
    #             fo.write(f"{n_at}\n")
    #             fo.write(f"Step {step} Image {i} "
    #                     f"Energy = {energies[i]:.10f}\n")
    #             for n in range(n_at):
    #                 sym = elem.elements[self.atomic_numbers[n]][0]
    #                 fo.write(f" {sym:2s} {cart_ang[n,0]:16.10f} "
    #                         f"{cart_ang[n,1]:16.10f} "
    #                         f"{cart_ang[n,2]:16.10f}\n")








    # def load_neb_path(self, source, n_resample=None, align=False, 
    #                   masses=None, verbose=True):
    #     """
    #     Load an existing reaction path as starting point for run_neb().
        
    #     Reads endpoint structures and intermediate images from various 
    #     formats, aligns them, converts to mass-weighted coordinates, and 
    #     optionally resamples to a different number of images. The result 
    #     is stored in self.neb_images_mw, which run_neb() picks up as a 
    #     warm start.
        
    #     This is the recommended workflow when you already have a Cartesian 
    #     NEB or IRC path and want to re-optimize it in mass-weighted space:
        
    #     Example:

    #         wkb.load_neb_path('previous_neb.xyz', n_resample=20)
    #         wkb.run_neb()
        
    #     Parameters
    #     ----------
    #     source : str
    #         Path to the input. Accepted formats:
            
    #         - Multi-image .xyz file: standard XYZ trajectory with energies 
    #           on the comment line. Same format as extract_coord_energy_xyz().
              
    #         - Directory of .xyz files: individual images as 000.xyz, 
    #           001.xyz, etc. Sorted alphabetically.
              
    #         - Directory of ORCA .hess/.engrad files: the projection folder 
    #           format from make_projection_files_orca(). Images read from 
    #           .hess files (which contain geometry).
              
    #         - restart.json: from a previous run_neb() (in scratch_dir/).
            
    #     n_resample : int, optional
    #         Resample the path to this many images using cubic spline 
    #         interpolation along the mass-weighted arc length. If None, 
    #         uses the images as read.
            
    #         Resampling produces equal spacing in mass-weighted arc length, 
    #         which means denser coverage near light atoms with large 
    #         displacements (the tunneling hydrogen).
            
    #     align : bool
    #         Sequentially align images via Kabsch before converting to 
    #         mass-weighted coordinates. Default True. Disable only if the 
    #         path is already consistently aligned.
            
    #     masses : list or ndarray, optional
    #         Override atomic masses for isotopic substitution. If None, 
    #         uses standard masses from the element table.
            
    #     verbose : bool
    #         Print summary. Default True.
            
    #     Attributes Set
    #     --------------
    #     self.neb_images_mw : list of ndarray, each (N_atoms, 3)
    #         Mass-weighted coordinates for each image. Detected by 
    #         run_neb() as a warm start.
    #     self.atomic_numbers : ndarray
    #         Populated if not already set.
    #     self.masses : ndarray
    #         Populated if not already set, or overridden if masses given.
    #     """
    #     if verbose:
    #         print(f"Loading NEB path from: {source}")
        
    #     # Detect format and read
    #     if os.path.isfile(source):
    #         if source.endswith('.xyz'):
    #             all_coords_bohr = self._neb_read_multi_xyz(source, verbose)
    #         else:
    #             # Try as xyz (there may be a different name but same structure)
    #             # Could give rise to weird issues if file is not multimage xyz...
    #             all_coords_bohr = self._neb_read_multi_xyz(source, verbose)
    #     elif os.path.isdir(source):
    #         all_coords_bohr = self._neb_read_directory(source, verbose)
    #     else:
    #         raise FileNotFoundError(f"Source not found: {source}")
        
    #     n_frames = len(all_coords_bohr)
        
    #     # Override masses if requested (isotopic substitution)
    #     if masses is not None:
    #         self.masses = np.array(masses)
    #         if verbose:
    #             print(f"  Using custom masses (isotopic substitution).")
        
    #     if len(self.masses) == 0:
    #         raise ValueError(
    #             "Masses not set. The source did not contain mass info "
    #             "and self.masses is empty. Call elements.get_masses() first or "
    #             "pass masses= explicitly."
    #         )
        
    #     # Sequential alignment
    #     if align:
    #         aligned = [all_coords_bohr[0]]
    #         for i in range(1, n_frames):
    #             img_aligned, _, _ = geom.kabsch_alignment_cm(
    #                 all_coords_bohr[i], aligned[i - 1], self.masses
    #             )
    #             aligned.append(img_aligned)
    #         all_coords_bohr = aligned
    #         if verbose:
    #             print(f"  Aligned {n_frames} images (sequential Kabsch).")
        
    #     # Resample
    #     if n_resample is not None and n_resample != n_frames:
    #         all_coords_bohr = band._neb_resample_path(
    #             all_coords_bohr, n_resample
    #         )
    #         n_frames = n_resample
    #         if verbose:
    #             print(f"  Resampled to {n_frames} images.")
        
    #     # Convert to mass-weighted
    #     sqrt_m = np.sqrt(self.masses)[:, np.newaxis]
    #     self.neb_images_mw = [coords * sqrt_m for coords in all_coords_bohr]
        
    #     # printing verbose information
    #     if verbose:
    #         ds_total = sum(
    #             np.linalg.norm(
    #                 self.neb_images_mw[i+1] - self.neb_images_mw[i]
    #             )
    #             for i in range(n_frames - 1)
    #         )
    #         print(f"  Total MW path length: {ds_total:.4f} "
    #               f"bohr*amu^(1/2)")
    #         print(f"  {n_frames} images ready. ")








    # def _neb_read_multi_xyz(self, filename, verbose=True):
    #     """
    #     Read a multi-frame XYZ file into a list of coordinate arrays.
        
    #     Also populates self.atomic_numbers and self.masses if they are 
    #     not already set. Energies on the comment line are read into 
    #     self.energies if available.
        
    #     Parameters
    #     ----------
    #     filename : str
    #     verbose : bool
        
    #     Returns
    #     -------
    #     all_coords_bohr : list of ndarray, each (N_atoms, 3)
    #     """
    #     all_coords = []
    #     energies = []
    #     first_frame_an = []
        
    #     with open(filename) as fo:
    #         lines = fo.readlines()
        
    #     i = 0
    #     frame = 0
    #     while i < len(lines):
    #         line = lines[i].strip()
    #         if not line:
    #             i += 1
    #             continue
            
    #         try:
    #             n_atoms = int(line)
    #         except ValueError:
    #             i += 1
    #             continue
            
    #         # Comment line so trying to extract energy
    #         i += 1
    #         if i < len(lines):
    #             comment = lines[i].strip()
    #             energy = self._neb_parse_comment_energy(comment)
    #             energies.append(energy)
            
    #         # Atom lines
    #         frame_an = []
    #         frame_coords = []
    #         for _ in range(n_atoms):
    #             i += 1
    #             if i >= len(lines):
    #                 break
    #             parts = lines[i].split()
    #             sym = parts[0]
    #             an = elem.search_element(sym)
    #             if an is None:
    #                 # Could be a numeric atomic number
    #                 try:
    #                     an = int(sym)
    #                 except ValueError:
    #                     raise ValueError(
    #                         f"Unknown atom '{sym}' in frame {frame}"
    #                     )
    #             frame_an.append(an)
    #             frame_coords.append([
    #                 float(parts[1]), float(parts[2]), float(parts[3])
    #             ])
            
    #         all_coords.append(
    #             np.array(frame_coords) * const.ang_to_bohr
    #         )
            
    #         if frame == 0:
    #             first_frame_an = frame_an
            
    #         frame += 1
    #         i += 1
        
    #     # Populate class attributes if needed
    #     if len(self.atomic_numbers) == 0:
    #         self.atomic_numbers = np.array(first_frame_an)
    #     if len(self.masses) == 0:
    #         self.masses = elem.get_masses(self.atomic_numbers)
        
    #     # Store energies if parsed
    #     valid_energies = [e for e in energies if e is not None]
    #     if len(valid_energies) == len(all_coords):
    #         self.energies = np.array(energies)
        
    #     if verbose:
    #         print(f"  Read {frame} frames from {filename}")
        
    #     return all_coords








    # # Method is kinda flimsy but works well with our current formats, so...
    # def _neb_parse_comment_energy(self, comment):
    #     """
    #     Try to extract a Hartree energy from an XYZ comment line.
        
    #     Handles the following formats:
    #       'Energy = -123.456'          (our format + common convention)
    #       'blah blah blah blah blah -123.456'   (field index 5, ORCA output)
        
    #     Returns None if no energy found.
    #     """
    #     if not comment:
    #         return None
        
    #     parts = comment.split()
        
    #     # Pattern: 'Energy = VALUE' or 'E = VALUE'
    #     for k, token in enumerate(parts):
    #         if token.lower() in ('energy', 'e') and k + 1 < len(parts):
    #             next_tok = parts[k + 1]
    #             if next_tok == '=' and k + 2 < len(parts):
    #                 try:
    #                     return float(parts[k + 2])
    #                 except ValueError:
    #                     pass
    #             else:
    #                 try:
    #                     return float(next_tok)
    #                 except ValueError:
    #                     pass
        
    #     # Orca trj format: energy at index 5
    #     if len(parts) > 5:
    #         try:
    #             val = float(parts[5])
    #             if -10000 < val < 0:
    #                 return val
    #         except ValueError:
    #             pass
        
    #     # Fallback: any negative float in Hartree range ¯\_(՞ ◡ ՞)_/¯ 
    #     for token in parts:
    #         try:
    #             val = float(token)
    #             if -10000 < val < -1:
    #                 return val
    #         except ValueError:
    #             continue
        
    #     return None








    # def _neb_read_directory(self, dirpath, verbose=True):
    #     """
    #     Read a directory of NEB images.
        
    #     Tries two strategies:
        
    #     1. ORCA .hess files (from make_projection_files_orca output):
    #        reads geometry from the $atoms block. Paired .engrad files 
    #        provide energies.
           
    #     2. Individual .xyz files: sorted alphabetically.
        
    #     Parameters
    #     ----------
    #     dirpath : str
    #     verbose : bool
        
    #     Returns
    #     -------
    #     all_coords_bohr : list of ndarray, each (N_atoms, 3)
    #     """
    #     all_files = sorted(os.listdir(dirpath))
        
    #     # Strategy 1: .hess files (ORCA projection folder)
    #     hess_files = [f for f in all_files if f.endswith('.hess')]
    #     if hess_files:
    #         if verbose:
    #             print(f"  Found {len(hess_files)} .hess files in {dirpath}")
            
    #         all_coords = []
    #         energies = []
            
    #         for hf in sorted(hess_files):
    #             hess_path = os.path.join(dirpath, hf)
    #             masses, atoms, coords, _, _, _ = self.read_orca_data(
    #                 hess_path
    #             )
    #             all_coords.append(coords)  # already in bohr **
                
    #             # Try to get energy from .engrad files
    #             stem = os.path.splitext(hf)[0]
    #             engrad_path = os.path.join(dirpath, stem + '.engrad')
    #             if os.path.exists(engrad_path):
    #                 gradient, energy = self.orca_engrad_parser(engrad_path)
    #                 energies.append(energy)
    #             else:
    #                 energies.append(None)
            
    #         # Populate class attributes with the information from files
    #         if len(self.atomic_numbers) == 0:
    #             self.atomic_numbers = np.array([
    #                 elem.search_element(a) for a in atoms
    #             ])
    #         if len(self.masses) == 0:
    #             self.masses = np.array(masses)
            
    #         valid_e = [e for e in energies if e is not None]
    #         if len(valid_e) == len(all_coords):
    #             self.energies = np.array(energies)
            
    #         return all_coords
        
    #     # Strategy 2: .xyz files
    #     xyz_files = [f for f in all_files if f.endswith('.xyz')]
    #     if xyz_files:
    #         if verbose:
    #             print(f"  Found {len(xyz_files)} .xyz files in {dirpath}")
            
    #         all_coords = []
    #         for xf in sorted(xyz_files):
    #             # Read each xyz using the multi-image reader 
    #             # but it works for single files too
    #             xyz_path = os.path.join(dirpath, xf)
    #             frame_coords = self._neb_read_multi_xyz(
    #                 xyz_path, verbose=False
    #             )
    #             # Take the first (and usually only) image
    #             if frame_coords:
    #                 all_coords.append(frame_coords[0])
            
    #         return all_coords
        
    #     raise FileNotFoundError(
    #         f"No .hess or .xyz files found in {dirpath}"
    #     )








    # def _neb_read_restart(self, filepath, verbose=True):
    #     """
    #     Read a restart.json from a previous run_neb().
        
    #     Restores images in mass-weighted coordinates, converts back to 
    #     Cartesian so the standard load_neb_path pipeline (alignment, 
    #     resampling) can proceed.
        
    #     Parameters
    #     ----------
    #     filepath : str
    #     verbose : bool
        
    #     Returns
    #     -------
    #     all_coords_bohr : list of ndarray, each (N_atoms, 3)
    #     """
        
    #     with open(filepath) as f:
    #         data = json.load(f)
        
    #     images_mw = [np.array(img) for img in data['images_mw']]
        
    #     # We need masses to convert back to Cartesian
    #     # If masses aren't set yet, we can't proceed
    #     if len(self.masses) == 0:
    #         raise ValueError(
    #             "Cannot read restart without masses. Set self.masses "
    #             "or self.atomic_numbers + elements.get_masses() before loading."
    #         )
        
    #     sqrt_m = np.sqrt(self.masses)[:, np.newaxis]
    #     all_coords_bohr = [mw / sqrt_m for mw in images_mw]
        
    #     # Restore energies and history if present
    #     if 'energies' in data:
    #         self.energies = np.array(data['energies'])
        
    #     step = data.get('step', '?')
    #     if verbose:
    #         print(f"  Read restart from step {step} "
    #               f"({len(images_mw)} images)")
        
    #     return all_coords_bohr