import os
import json
import numpy as np
import pprint
from dataclasses import replace
from tunkit.core import elements as elem
from tunkit.core import constants as const
from tunkit.path import projections






def load_cluster_results(path,
                         json_file="tunneling_results.json", 
                         threshold=1.0E-4, 
                         diagnostic_path_freq=False, 
                        #  invert_curve=False, 
                         masses=None,
                         verbose=True):
    """Loads the pre-parsed cluster JSON file directly and processes the path."""

    
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
        return
        
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    # load data output
    energies = np.array(data["energies"])
    gradients = np.array(data["gradients"])
    hessians = np.array([np.array(h) for h in data["hessians"]])
    
    raw_coord_ang = np.array(data["coordinates"])
    raw_coord = raw_coord_ang * const.ang_to_bohr
    proj_coordinates = raw_coord 
    

    # Reverse path direction before any processing (same as invert_curve in NEB extraction)
    # if invert_curve:
    #     self.energies = self.energies[::-1]
    #     self.gradients = self.gradients[::-1]
    #     self.hessians = self.hessians[::-1]
    #     self.raw_coord_ang = self.raw_coord_ang[::-1]
    #     self.raw_coord = self.raw_coord[::-1]
    #     self.proj_coordinates = self.proj_coordinates[::-1]


    symbols = data["atomic_symbols"]
    at_nums = []
    for sym in symbols:
        num = elem.atomic_number_from_symbol(sym)
        if num:
            at_nums.append(num)
        else:
            print(f"Warning: Could not identify atomic number for {sym}")
            at_nums.append(0)
    atomic_numbers = np.array(at_nums)
    
    if masses is None:
        masses = []
        masses = elem.get_masses(atomic_numbers)
    else:
        masses = np.array(masses)
        if len(masses) != len(atomic_numbers):
            raise ValueError("Error: Length of masses array does not match number of atoms.")
            
    if verbose:
        print(f"Successfully loaded {len(energies)} raw steps from {json_file}!")

    out = projections.process_path_data(proj_coordinates,
                                        hessians,
                                        masses,
                                        energies,
                                        threshold=threshold,  
                                        diagnostic_path_freq=diagnostic_path_freq, 
                                        verbose=verbose)

    return replace(path,
                   hessians=hessians,
                   proj_coordinates=proj_coordinates,
                   gradients=gradients,
                   atomic_numbers=atomic_numbers,
                   raw_coord=raw_coord,
                   raw_coord_ang=raw_coord_ang,
                   energies=energies,
                   masses=masses,
                   gradient_rms=out['gradient_rms'],
                   projected_corrections=out['projected_corrections'],
                   mw_hessians=out['mw_hessians'],
                   normal_modes=out['normal_modes'],
                   mw_gradients=out['mw_gradients'],
                   path_frequencies=out['path_frequencies'],
                   tangents=out['tangents'],
                   projected_frequencies=out['projected_frequencies'],
                   proj_energies=out['proj_energies'])








def generate_projection_payload(
    path, 
    script_name='run_projections.py',
    results_file_name='tunneling_results',
    method='B3LYP', 
    basis_set='', 
    charge=0,
    multiplicity=1, 
    num_procs=4, 
    mem='1000',
    extra_options='', 
    orca_path='orca',
    band_xyz_file_source=None,
    n_parallel=4
):
    """Generates the script for the driver to run projections in the cluster."""
    
    # Formating the coordinates and symbols
    symbols = [elem.elements[int(an)][0] for an in path.atomic_numbers]
    if path.proj_coordinates is not None:
        coords = path.proj_coordinates * const.bohr_to_ang # proj_coordinates is in bohr
        coords_list = pprint.pformat(coords.tolist(), width=80)
    else:
        coords_list = pprint.pformat(path.raw_coord_ang.tolist(), width=80)
    
    # Building the ORCA header string
    header_string = f"""! freq engrad tightscf 
!{method} {basis_set}
{extra_options}
%maxcore {mem}
%pal nprocs {num_procs} end
* xyz {charge} {multiplicity}
"""

    # Generating the Python file
    # First part of the file are all the parameters, so that they can be 
    # edited with easy and the file can be reused if needed
    script_content = f'''#!/usr/bin/env python3
"""
Projection Driver Script
Upload this file to the cluster and run it within the virtual environment.
"""
import os
import json
import tunkit.core.elements as elem 
import tunkit.io.orca as orca_io 
import tunkit.io.ee_inputs as ee_inputs
import tunkit.core.engine as engine
from tunkit.path.reaction_path import ReactionPath

#*************************************************
#*                Run settings                   *
#*************************************************

# ORCA needs the full path to work
ORCA_EXE = {repr(orca_path)}
N_PARALLEL = {n_parallel}
SCRATCH_DIR = "projections_scratch"

ORCA_HEADER = """{header_string.strip()}
"""

RESULTS_FILE_NAME = {results_file_name!r}

# If set to a string (e.g., 'neb_MEP_trj.xyz')
# coordinates will be read from the file. If set to None,
# the coordinates com from the COORDINATES list.
SOURCE_XYZ_FILE = {repr(band_xyz_file_source)}

ATOMIC_SYMBOLS = {symbols}
COORDINATES = {coords_list}

#*************************************************
#*  Driver (don't modify it if you're not sure)  *
#*************************************************

def main():

    print("Generating ORCA inputs...")

    # Mapping the symbols to atomic_numbers
    atomic_numbers = [elem.atomic_number_from_symbol(s) for s in ATOMIC_SYMBOLS]

    if SOURCE_XYZ_FILE is not None and SOURCE_XYZ_FILE != 'None':
        assert os.path.exists(SOURCE_XYZ_FILE), (f"ERROR: the file '{{SOURCE_XYZ_FILE}}' "
            "could not be found! Check if SOURCE_XYZ_FILE variable is set correctly or "
            "if it is set to None.")
            
        print(f"Warning: Loading coordinates from {{SOURCE_XYZ_FILE}}")
        temp_path = ReactionPath()
        temp_path = orca_io.extract_coord_energy_xyz(temp_path, SOURCE_XYZ_FILE, verbose=False)
        coordinates = temp_path.raw_coord_ang.tolist()
        atomic_numbers = temp_path.atomic_numbers.tolist()
        atomic_symbols = [elem.elements[int(an)][0] for an in temp_path.atomic_numbers]
    else:
        print("Using coordinates found embedded in the payload file.")
        coordinates = COORDINATES
        atomic_symbols = ATOMIC_SYMBOLS

    assert coordinates, ("ERROR: Couldn't load the band from '{{SOURCE_XYZ_FILE}}', "
                          "or no embedded coordinates found. Check the "
                          "SOURCE_XYZ_FILE variable (None means read "
                          "from file and check the embedded coordinates")

    # Writing the input files
    scratch_path = ee_inputs.make_projection_files_orca(
        coord_ang=coordinates, 
        atomic_numbers=atomic_numbers, 
        path_to_folder=SCRATCH_DIR,
        header=ORCA_HEADER, 
        verbose=True
    )
    
    # Preparing the input for engine.run_orca_parallel (idx, filepath)
    inputs = []
    for num in range(len(coordinates)):
        filename = os.path.join(scratch_path, f"{{num:03d}}.inp")
        inputs.append((num, filename))
        
    print(f"Running {{len(inputs)}} ORCA jobs in parallel (Max {{N_PARALLEL}})...")

    failed = engine.run_orca_parallel(ORCA_EXE, inputs, N_PARALLEL)
    
    if failed:
        print(f"WARNING! The following jobs failed: {{failed}}")
        
    print("Extracting projection data...")

    parsed_data = {{
        "atomic_symbols": atomic_symbols,
        "coordinates": coordinates,
        "energies": [],
        "gradients": [],
        "hessians": []
    }}
    
    for num, _ in inputs:
        stem = os.path.join(scratch_path, f"{{num:03d}}")
        engrad_file = stem + ".engrad"
        hess_file = stem + ".hess"
        
        # checking error files
        if not os.path.exists(engrad_file) or not os.path.exists(hess_file):
            print(f"WARNING: Missing results for step {{num}}. Check ORCA output.")
            continue
            
        gradient, energy = orca_io.orca_engrad_parser(engrad_file)
        _, _, _, hessian, _, _ = orca_io.read_orca_data(hess_file)
        
        parsed_data["energies"].append(energy)
        parsed_data["gradients"].append(gradient)
        # read_orca_data returns hessian as a numpy array, so we must convert to list for JSON
        parsed_data["hessians"].append(hessian.tolist()) 
        
    output_json = RESULTS_FILE_NAME + ".json"
    with open(output_json, "w") as f:
        json.dump(parsed_data, f)
        
    print(f"Done! Download {{output_json}} to your local machine.")

if __name__ == "__main__":
    main()

'''
    with open(script_name, 'w') as f:
        f.write(script_content)
    print(f"Successfully generated {script_name}!")








def generate_neb_driver(
    self,
    script_name='run_neb_cluster.py',
    method='B3LYP',
    basis_set='def2-SVP',
    charge=0,
    multiplicity=1,
    num_procs=4,
    mem='1000',
    extra_options='',
    orca_path='orca',
    n_images=15,
    k_spring=0.1,
    ts_xyz_path=None,
    ts_mode='frozen',
    max_steps=300,
    optimizer='vpo',
    n_parallel=4,
    job_prefix='neb_run',
    source_file=None
):
    """Generates a clean Driver Script for the cluster to run the NEB optimization."""
    
    # 1. Format the coordinates and symbols
    symbols = [elem.elements[int(an)][0] for an in self.atomic_numbers]
    
    if source_file is None:
        # Cold start: Extract endpoints
        if len(self.raw_coord_ang) == 0:
            raise ValueError("No coordinates loaded. Either load a structure first or pass source_file.")
        reactant_coords = pprint.pformat(self.raw_coord_ang[0].tolist(), width=80)
        product_coords = pprint.pformat(self.raw_coord_ang[-1].tolist(), width=80)
    else:
        reactant_coords = "None"
        product_coords = "None"

    # Extract TS coordinates if provided
    if ts_mode and ts_xyz_path:
        ts_coords_array = self._neb_read_multi_xyz(ts_xyz_path, verbose=False)[0]
        ts_coords = pprint.pformat(ts_coords_array.tolist(), width=80)
    else:
        ts_coords = "None"
        ts_mode = None

    # 2. Build the standard ORCA header string (engrad only, no freq)
    header_string = f"""! engrad tightscf 
!{method} {basis_set}
{extra_options}
%maxcore {mem}
%pal nprocs {num_procs} end
* xyz {charge} {multiplicity}
"""

    # 3. Generate the editable Python file
    script_content = f'''#!/usr/bin/env python3
"""
NEB Driver Script
Upload this file to the cluster and run it within your virtual environment.
"""
import os
import numpy as np
from tunkit import WKB_Tunneling_Analysis
from tunkit.core.elements import atomic_number_from_symbol, get_masses

# ==========================================
# 1. USER SETTINGS (Editable on the cluster)
# ==========================================
ORCA_EXE = {repr(orca_path)}
N_PARALLEL = {n_parallel}
SCRATCH_DIR = "neb_scratch"
JOB_PREFIX = {repr(job_prefix)}

# NEB Hyperparameters
N_IMAGES = {n_images}
K_SPRING = {k_spring}
OPTIMIZER = {repr(optimizer)}
MAX_STEPS = {max_steps}
TS_MODE = {repr(ts_mode)}

# Theory Level
ORCA_HEADER = """{header_string.strip()}
"""

# Geometries (Angstrom)
ATOMIC_SYMBOLS = {symbols}
REACTANT_COORDS = {reactant_coords}
PRODUCT_COORDS = {product_coords}
TS_COORDS = {ts_coords}

# Warm Start Path (Set this to a filename string if you want to load an existing path)
SOURCE_FILE = {repr(source_file)}

# ==========================================
# 2. EXECUTION LOGIC
# ==========================================
def main():
    print(f"Starting NEB calculation: {{JOB_PREFIX}}")
    
    # Initialize the main analysis object
    wkb = WKB_Tunneling_Analysis()
    wkb.atomic_numbers = np.array([atomic_number_from_symbol(s) for s in ATOMIC_SYMBOLS])
    wkb.masses = get_masses(wkb.atomic_numbers)
    wkb.neb_header = ORCA_HEADER
    
    # Convert lists to numpy arrays if they exist
    R_coords = np.array(REACTANT_COORDS) if REACTANT_COORDS is not None else None
    P_coords = np.array(PRODUCT_COORDS) if PRODUCT_COORDS is not None else None
    T_coords = np.array(TS_COORDS) if TS_COORDS is not None else None

    # Load warm start if requested
    if SOURCE_FILE:
        print(f"Loading warm start path from {{SOURCE_FILE}}...")
        wkb.load_neb_path(SOURCE_FILE)

    # Run the NEB optimization
    wkb.run_neb(
        reactant_coords=R_coords,
        product_coords=P_coords,
        ts_coords=T_coords,
        ts_mode=TS_MODE,
        n_images=N_IMAGES,
        k_spring=K_SPRING,
        max_steps=MAX_STEPS,
        optimizer=OPTIMIZER,
        orca_path=ORCA_EXE,
        n_parallel=N_PARALLEL,
        scratch_dir=SCRATCH_DIR,
        job_prefix=JOB_PREFIX,
        verbose=True
    )
    
    print("\\nNEB finished! Download the scratch directory for results.")

if __name__ == "__main__":
    main()
'''
    with open(script_name, 'w') as f:
        f.write(script_content)
    print(f"Successfully generated {script_name}!")










# class ClusterIOMixin:

#     def generate_cluster_payload(self, script_name='run_tunneling_orca.py', output_name=None):
#         """Generates a standalone Python script to run and parse ORCA jobs on a cluster."""
#         import pprint
#         if len(self.raw_coord_ang) == 0:
#             print("Error: No coordinates loaded. Please extract coordinates first.")
#             return

#         # Prepare the Data Section
#         symbols = [elem.elements[int(an)][0] for an in self.atomic_numbers]
#         coords_list = self.raw_coord_ang.tolist()
#         coords_list = pprint.pformat(coords_list, width=60) # be careful if the precision is raised, 
#                                                             # as the width would have to be raised too
        
#         #output name
#         if output_name is None:
#             output_name = os.path.splitext(os.path.basename(script_name))[0] + '.json'

#         # Using a raw string for the static parts to avoid f-string curly brace clashes
#         script_content = f'''#!/usr/bin/env python3
# # --- GENERATED DATA SECTION ---
# import os
# import subprocess
# import json
# import sys
# from concurrent.futures import ThreadPoolExecutor

# ORCA_HEADER = """{self.header.strip()}
# """
# ATOMIC_SYMBOLS = {symbols}
# COORDINATES = {coords_list}

# # --- STATIC ENGINE SECTION ---

# def write_inputs(scratch_dir):
#     os.makedirs(scratch_dir, exist_ok=True)
#     inputs = []
#     for step_idx, step_coords in enumerate(COORDINATES):
#         filename = os.path.join(scratch_dir, f"step_{{step_idx:03d}}.inp")
#         with open(filename, "w") as f:
#             f.write(ORCA_HEADER + "\\n")
#             for atom_sym, (x, y, z) in zip(ATOMIC_SYMBOLS, step_coords):
#                 f.write(f" {{atom_sym}} {{x:.6f}} {{y:.6f}} {{z:.6f}}\\n")
#             f.write("*\\n")
#         inputs.append(filename)
#     return inputs

# def run_job(orca_exe, inp_file):
#     scratch_dir = os.path.dirname(inp_file)
#     out_file = inp_file.replace(".inp", ".out")
#     print(f"Running {{os.path.basename(inp_file)}}...")
#     with open(out_file, "w") as outfile:
#         subprocess.run([orca_exe, os.path.basename(inp_file)], cwd=scratch_dir, stdout=outfile, stderr=subprocess.STDOUT)

# def extract_engrad(filepath):
#     gradient = []
#     energy = 0.0
#     with open(filepath, 'r') as f:
#         nblocks = 0
#         for line in f:
#             if '#' not in line:
#                 nblocks += 1
#             if nblocks >= 3 and '#' in line:
#                 break
#             if nblocks >= 3:
#                 gradient.append(float(line.strip()))
#                 continue
#             if nblocks == 2 and '#' not in line:
#                 energy = float(line.strip())
#     return energy, gradient

# def extract_hessian(filepath):
#     hessian = []
#     with open(filepath, 'r') as f:
#         lines = f.readlines()
        
#     i = 0
#     while i < len(lines):
#         line = lines[i].strip()
#         if line.startswith("$hessian"):
#             dim = int(lines[i+1].strip())
#             hessian = [[0.0] * dim for _ in range(dim)]
#             i += 2
#             current_cols = []
#             while i < len(lines):
#                 curr = lines[i].strip()
#                 if curr.startswith("$"): 
#                     break 
#                 if not curr or curr.startswith("#"):
#                     i += 1
#                     continue
                
#                 parts = curr.split()
#                 is_data_line = False
#                 if len(parts) > 1 and ("." in parts[1] or "E" in parts[1] or "e" in parts[1]):
#                     is_data_line = True
                    
#                 if not is_data_line:
#                     current_cols = [int(c) for c in parts if c.lstrip('-').isdigit()]
#                 else:
#                     row_idx = int(parts[0])
#                     vals = [float(x) for x in parts[1:]]
#                     for k, val in enumerate(vals):
#                         if k < len(current_cols):
#                             hessian[row_idx][current_cols[k]] = val
#                 i += 1
#             break 
#         i += 1
#     return hessian

# def extract_data(scratch_dir):
#     results = {{
#         "atomic_symbols": ATOMIC_SYMBOLS,
#         "coordinates": COORDINATES,
#         "energies": [], 
#         "gradients": [], 
#         "hessians": []
#     }}
    
#     for step_idx in range(len(COORDINATES)):
#         base = os.path.join(scratch_dir, f"step_{{step_idx:03d}}")
#         engrad_file = base + ".engrad"
#         hess_file = base + ".hess"
        
#         if not os.path.exists(engrad_file) or not os.path.exists(hess_file):
#             print(f"WARNING: Missing results for step {{step_idx}}. Check ORCA output.")
#             continue
            
#         energy, gradient = extract_engrad(engrad_file)
#         hessian = extract_hessian(hess_file)
                    
#         results["energies"].append(energy)
#         results["gradients"].append(gradient)
#         results["hessians"].append(hessian)
        
#     return results

# def main():
#     if len(sys.argv) < 3:
#         print("Usage: python run_tunneling_orca.py <path_to_orca> <n_concurrent_jobs>")
#         sys.exit(1)
        
#     orca_exe = sys.argv[1]
#     n_jobs = int(sys.argv[2])
#     scratch_dir = "tunneling_scratch"
    
#     inputs = write_inputs(scratch_dir)
    
#     with ThreadPoolExecutor(max_workers=n_jobs) as executor:
#         for inp in inputs:
#             executor.submit(run_job, orca_exe, inp)
            
#     print("All jobs finished. Extracting data...")
#     parsed_data = extract_data(scratch_dir)
    
#     with open("{output_name}", "w") as f:
#         json.dump(parsed_data, f)
        
#     print("Done! Download {output_name} to your local machine.")

# if __name__ == "__main__":
#     main()
# '''
#         with open(script_name, 'w') as f:
#             f.write(script_content)
            
#         print(f"Successfully generated {script_name}! Upload this file to your cluster.")








#     def generate_projection_payload(self,
#                                 script_name='run_projection_payload.py',
#                                 method='', basis_set='', charge=0,
#                                 multiplicity=1, num_procs=4, mem='1000',
#                                 extra_options='', orca_path='orca',
#                                 n_parallel=4):
#         """
#     Emit a standalone post-NEB PAYLOAD script for the cluster.

#     Takes the ORCA NEB band file as argv[1] (NEB .out optional as argv[2]),
#     verifies the NEB terminated normally, builds freq+engrad projection
#     inputs along the band, runs them in the projections scratch folder, and
#     writes <band>_proj.json with energies/gradients/hessians/coordinates/
#     atomic_symbols. Download it and load locally with load_cluster_results().
#     It runs no downstream analysis. Imports WKB_Tunneling_Analysis, so upload
#     it next to tunnel_code.py and call it after the ORCA NEB step in SLURM.
#     """
#         script = f'''#!/usr/bin/env python3
# """Post-NEB projection PAYLOAD. Generated by generate_projection_payload().

# Runs the freq+engrad projections along an ORCA NEB band and writes a JSON
# (energies/gradients/hessians/coordinates/atomic_symbols) to download and load
# locally with load_cluster_results(). Does NOT run the downstream analysis.

# Usage (in your SLURM file, right after the ORCA NEB call):
#     orca neb.inp > neb.out
#     python {script_name} neb_MEP_trj.xyz neb.out
# """
# import WKB_Tunneling_Analysis as tc
# import os
# import subprocess
# import json
# import sys
# from concurrent.futures import ThreadPoolExecutor

# orca_exe  = {orca_path!r}
# N_PARALLEL = {n_parallel}
# METHOD     = {method!r}
# BASIS      = {basis_set!r}
# CHARGE     = {charge}
# MULT       = {multiplicity}
# NUM_PROCS  = {num_procs}
# MEM        = {mem!r}
# EXTRA      = {extra_options!r}


# def orca_ok(out_path, tail_bytes=8192):
#     """True iff 'ORCA TERMINATED NORMALLY' appears near the end of out_path."""
#     try:
#         with open(out_path, "rb") as f:
#             f.seek(0, os.SEEK_END)
#             size = f.tell()
#             f.seek(max(0, size - tail_bytes))
#             tail = f.read().decode("utf-8", "ignore")
#     except OSError:
#         return False
#     return "ORCA TERMINATED NORMALLY" in tail

    
# def run_job(orca_exe, inp_file):
#     scratch_dir = os.path.dirname(inp_file)
#     out_file = inp_file.replace(".inp", ".out")
#     print(f"Running {{os.path.basename(inp_file)}}...")
#     with open(out_file, "w") as outfile:
#         subprocess.run([orca_exe, os.path.basename(inp_file)], cwd=scratch_dir, stdout=outfile, stderr=subprocess.STDOUT)

        
# def extract_engrad(filepath):
#     gradient = []
#     energy = 0.0
#     with open(filepath, "r") as f:
#         nblocks = 0
#         for line in f:
#             if "#" not in line:
#                 nblocks += 1
#             if nblocks >= 3 and "#" in line:
#                 break
#             if nblocks >= 3:
#                 gradient.append(float(line.strip()))
#                 continue
#             if nblocks == 2 and "#" not in line:
#                 energy = float(line.strip())
#     return energy, gradient


# def extract_hessian(filepath):
#     hessian = []
#     with open(filepath, 'r') as f:
#         lines = f.readlines()
        
#     i = 0
#     while i < len(lines):
#         line = lines[i].strip()
#         if line.startswith("$hessian"):
#             dim = int(lines[i+1].strip())
#             hessian = [[0.0] * dim for _ in range(dim)]
#             i += 2
#             current_cols = []
#             while i < len(lines):
#                 curr = lines[i].strip()
#                 if curr.startswith("$"): 
#                     break 
#                 if not curr or curr.startswith("#"):
#                     i += 1
#                     continue
                
#                 parts = curr.split()
#                 is_data_line = False
#                 if len(parts) > 1 and ("." in parts[1] or "E" in parts[1] or "e" in parts[1]):
#                     is_data_line = True
                    
#                 if not is_data_line:
#                     current_cols = [int(c) for c in parts if c.lstrip('-').isdigit()]
#                 else:
#                     row_idx = int(parts[0])
#                     vals = [float(x) for x in parts[1:]]
#                     for k, val in enumerate(vals):
#                         if k < len(current_cols):
#                             hessian[row_idx][current_cols[k]] = val
#                 i += 1
#             break 
#         i += 1
#     return hessian


# def extract_data(scratch_dir, atomic_symbols, coordinates):
#     results = {{
#         "atomic_symbols": atomic_symbols,
#         "coordinates": coordinates,
#         "energies": [], 
#         "gradients": [], 
#         "hessians": []
#     }}
    
#     for step_idx in range(len(coordinates)):
#         base = os.path.join(scratch_dir, f"step_{{step_idx:03d}}")
#         engrad_file = base + ".engrad"
#         hess_file = base + ".hess"
        
#         if not os.path.exists(engrad_file) or not os.path.exists(hess_file):
#             sys.exit("ABORT: Missing results for step {{step_idx}}. Projection(s) failed/incomplete.")
            
#         energy, gradient = extract_engrad(engrad_file)
#         hessian = extract_hessian(hess_file)
                    
#         results["energies"].append(energy)
#         results["gradients"].append(gradient)
#         results["hessians"].append(hessian)
        
#     return results


# def write_inputs(scratch_dir, coordinates, orca_header, atomic_symbols):
#     os.makedirs(scratch_dir, exist_ok=True)
#     inputs = []
#     for step_idx, step_coords in enumerate(coordinates):
#         filename = os.path.join(scratch_dir, f"step_{{step_idx:03d}}.inp")
#         with open(filename, "w") as f:
#             f.write(orca_header + "\\n")
#             for atom_sym, (x, y, z) in zip(atomic_symbols, step_coords):
#                 f.write(f" {{atom_sym}} {{x:.6f}} {{y:.6f}} {{z:.6f}}\\n")
#             f.write("*\\n")
#         inputs.append(filename)
#     return inputs


# def main():
#     if len(sys.argv) < 3:
#         sys.exit("Usage: python " + sys.argv[0] + " <band_file.xyz> <n_concurrent_jobs>")
#     band_file = sys.argv[1]
#     neb_out   = sys.argv[1].removesuffix('_MEP_trj.xyz') + '.out'
#     #orca_exe = sys.argv[2]
#     n_jobs = int(sys.argv[2])
#     scratch_dir = "tunneling_scratch"

#     json_file = sys.argv[1].removesuffix('_MEP_trj.xyz') + '_projected.json'
    
#     print('Checking NEB success on file:', neb_out)

#     # 1. Gate on NEB completion
#     if neb_out is not None and not orca_ok(neb_out):
#         sys.exit("ABORT: no 'ORCA TERMINATED NORMALLY' in " + neb_out)
    
#     print('NEB terminated succesfully!')    
    
#     print('Loading NEB from file: ', band_file)
#     if not os.path.isfile(band_file):
#         sys.exit("ABORT: band file not found: " + band_file)

#     # 2. Load band + build freq+engrad inputs in the projections scratch folder
#     wkb = tc.WKB_Tunneling_Analysis()
#     wkb.mep_file_name = band_file
#     wkb.extract_coord_energy_xyz()
#     wkb.make_header_orca(method=METHOD, basis_set=BASIS, charge=CHARGE,
#                          multiplicity=MULT, num_procs=NUM_PROCS, mem=MEM,
#                          extra_options=EXTRA, make_header_neb=False)
    
#     orca_header = wkb.header.strip()
#     atomic_symbols = [wkb.elements[int(an)][0] for an in wkb.atomic_numbers]
#     coordinates = wkb.raw_coord_ang.tolist()

#     inputs = write_inputs(scratch_dir, coordinates, orca_header, atomic_symbols)

#     with ThreadPoolExecutor(max_workers=n_jobs) as executor:
#         for inp in inputs:
#             executor.submit(run_job, orca_exe, inp)

#     print("All jobs finished. Extracting data...")
#     parsed_data = extract_data(scratch_dir, atomic_symbols, coordinates)

#     with open(json_file, "w") as f:
#         json.dump(parsed_data, f)


# if __name__ == "__main__":
#     main()
# '''
#         with open(script_name, 'w') as f:
#             f.write(script)
#         print(f"Wrote {script_name}. Upload it next to tunnel_code.py and call it "
#             f"after the ORCA NEB step in your SLURM file.")









#     #[TODO] Just like the run_neb method, we should take the ts_mode off as it 
#     # was not helping convergence or avoiding kinks
#     def generate_neb_script(self, 
#                                 script_name='run_neb.py',
#                                 method='',
#                                 basis_set='',
#                                 charge=0,
#                                 multiplicity=1,
#                                 num_procs=4,
#                                 mem='1000',
#                                 extra_options='',
#                                 orca_path='orca',
#                                 n_images=15,
#                                 k_spring=0.1,
#                                 ts_xyz_path=None,
#                                 ts_mode='frozen',
#                                 validate_ts_gradient=False,
#                                 max_steps=300,
#                                 fmax_tol=450e-6, 
#                                 frms_tol=300e-6,
#                                 optimizer='vpo',
#                                 scope='global',
#                                 split_forces=False,
#                                 precondition=False,
#                                 max_move=0.2,
#                                 lbfgs_mem=20,
#                                 lbfgs_dr=1.0e-3,
#                                 n_parallel=4,
#                                 scratch_dir='neb_scratch',
#                                 job_prefix=None,
#                                 save_every=10,
#                                 skip_alignment=False,
#                                 align_every_step=False,
#                                 source_file=None,
#                                 n_resample=None,
#                                 verbose=True):
#             """
#             Generate a Python script that runs the MW-NEB optimization.
            
#             The generated script is self-contained: it imports 
#             WKB_Tunneling_Analysis, sets up the header, loads the path or 
#             endpoints, and calls run_neb(). The user uploads it to the 
#             cluster alongside tunnel_code.py and submits it through their 
#             scheduler.
            
#             Parameters
#             ----------
#             script_name : str
#                 Output filename. Default 'run_neb.py'.
#             method : str
#                 Electronic structure method (e.g., 'B3LYP').
#             basis_set : str
#                 Basis set (e.g., 'def2-SVP').
#             charge, multiplicity : int
#                 Molecular charge and spin multiplicity.
#             num_procs : int
#                 ORCA processors per image. Default 4.
#             mem : str
#                 ORCA memory per core in MB. Default '4000'.
#             extra_options : str
#                 Additional ORCA keywords (e.g., '! RIJCOSX def2/J').
#             orca_path : str
#                 Path to ORCA on the cluster. Default 'orca'.
#             n_images : int
#                 Number of NEB images. Default 15.
#             k_spring : float
#                 Spring constant. Default 0.1.
#             ts_xyz_path : str, optional
#                 Path to a single-frame XYZ file with the TS geometry.
#                 Required if ts_mode is not None.
#             ts_mode : str or None
#                 'frozen' (default), 'free', or None (plain NEB, no TS).
#             validate_ts_gradient : bool
#                 Opt-in TS gradient check. Default False.
#             max_steps : int
#                 Maximum iterations. Default 300.
#             fmax_tol, frms_tol : float
#                 Force convergence thresholds (Eh/bohr).
#             optimizer : str
#                 'vpo' (default), 'fire', or 'lbfgs'.
#             scope : str
#                 'global' (default) or 'local'. Ignored for lbfgs.
#             split_forces : bool
#                 Separate F_perp and F_spring for the optimizer. Default False.
#             precondition : bool
#                 Mass preconditioning on F_perp. Default False.
#             max_move : float
#                 Trust-radius cap. Default 0.2.
#             lbfgs_mem, lbfgs_dr : float
#                 L-BFGS-specific parameters.
#             n_parallel : int
#                 Concurrent ORCA jobs. Default 4.
#             scratch_dir : str
#                 Working directory for ORCA files. Default 'neb_scratch'.
#             save_every : int
#                 Save restart every N steps. Default 10.
#             skip_alignment : bool
#                 Skip Kabsch alignment on warm-start reload. Default False.
#             source_file : str, optional
#                 Warm start from a previous NEB trajectory.
#             n_resample : int, optional
#                 Resample loaded path to this many images.
#             verbose : bool
#                 Print info. Default True.
#             """
#             import pprint

#             if ts_mode == 'None': ts_mode = None

#             if source_file == 'None': source_file = None

#             script_name_loader = ''
#             if job_prefix == None:
#                 job_prefix = "script_name"
#                 script_name_loader = "script_name = sys.argv[0].removesuffix('.py')"

#             # Build the coordinate block if not using a source file
#             if source_file is None:
#                 if len(self.raw_coord_ang) == 0:
#                     raise ValueError(
#                         "No coordinates loaded. Either load a structure "
#                         "first or pass source_file= for a warm start."
#                     )
                
#                 symbols = [
#                     elem.elements[int(an)][0] for an in self.atomic_numbers
#                 ]

#                 # all frames embedded in the payload,
#                 # raise width if the precision is increased as the width will change
#                 coords_list = pprint.pformat(
#                     self.raw_coord_ang.tolist(), width=60
#                 )
            
            

#                 coord_block = f"""
# # --- Geometries (Angstrom) ---
# # COORDINATES holds the full embedded path:
# #   2 frames      -> cold start, interpolate a band between the endpoints
# #   >2 frames     -> warm start, use the embedded frames as the initial band
# SYMBOLS = {symbols}
# COORDINATES = {coords_list}
# _SKIP_ALIGNMENT = {skip_alignment}
# _N_RESAMPLE = {n_resample!r}
# """
#                 # [TODO] The sequential alignement is exposed and it should be its own
#                 # method to help maintenance and tidyness. Also load_neb_path and 
#                 # _resample_path are not well thought out.
#                 load_block = """
# # --- Load embedded geometries ---
# import numpy as np
# import sys
# wkb.atomic_numbers = np.array([wkb.search_element(s) for s in SYMBOLS])
# wkb.get_masses()
# wkb.raw_coord_ang = np.array(COORDINATES)
# wkb.raw_coord = wkb.raw_coord_ang * wkb.ang_to_bohr

# if len(wkb.raw_coord) > 2:
#     # Embedded multi-image path -> warm start from this band.
#     _frames = [c for c in wkb.raw_coord]          # bohr, per image

#     # Sequential mass-weighted Kabsch alignment (same as load_neb_path),
#     # unless alignment was explicitly skipped.
#     if not _SKIP_ALIGNMENT:
#         _aligned = [_frames[0]]
#         for _i in range(1, len(_frames)):
#             _img, _, _ = wkb.kabsch_alignment_cm(_frames[_i], _aligned[_i - 1])
#             _aligned.append(_img)
#         _frames = _aligned
#         print(f"  Aligned {len(_frames)} images (sequential Kabsch).")
#     else:
#         print("  Skipping alignment of embedded band (skip_alignment=True).")

#     # Optional resample along the mass-weighted arc length.
#     if _N_RESAMPLE is not None and _N_RESAMPLE != len(_frames):
#         _frames = wkb._neb_resample_path(_frames, _N_RESAMPLE)
#         print(f"  Resampled to {len(_frames)} images.")

#     _sqrt_m = np.sqrt(wkb.masses)[:, np.newaxis]
#     wkb.neb_images_mw = [c * _sqrt_m for c in _frames]
#     REACTANT_COORDS = None
#     PRODUCT_COORDS = None
# else:
#     # Cold start: run_neb interpolates a band between the two endpoints.
#     REACTANT_COORDS = wkb.raw_coord[0]
#     PRODUCT_COORDS = wkb.raw_coord[-1]
# """
#             else:
#                 coord_block = f"""
# # --- Source file for warm start ---
# SOURCE_FILE = '{source_file}'
# """
#                 resample_line = (
#                     f", n_resample={n_resample}" if n_resample else ", n_resample=None"
#                 )
#                 align_line = ", align=False" if skip_alignment else ", align=True"
#                 load_block = f"""
# # --- Load existing path ---
# wkb.load_neb_path(SOURCE_FILE{resample_line}{align_line})
# REACTANT_COORDS = None
# PRODUCT_COORDS = None
# """

#             # TS block (it is conditional on ts_mode)
#             if ts_mode is not None:
#                 if ts_xyz_path is None:
#                     raise ValueError(
#                         f"ts_mode='{ts_mode}' requires ts_xyz_path."
#                     )
#                 ts_block = f"""
# # --- Load TS geometry ---
# _ts_wkb = WKB_Tunneling_Analysis()
# _ts_wkb.atomic_numbers = wkb.atomic_numbers
# _ts_wkb.masses = wkb.masses
# _ts_coords_list = _ts_wkb._neb_read_multi_xyz({repr(ts_xyz_path)}, verbose=False)
# TS_COORDS = _ts_coords_list[0]  # bohr (read_multi_xyz already converts)
# """
#                 ts_arg = "ts_coords=TS_COORDS,"
#             else:
#                 ts_block = """
# # --- No TS supplied (plain NEB) ---
# TS_COORDS = None
# """
#                 ts_arg = "ts_coords=None,"

#             script_content = f'''#!/usr/bin/env python3
# """
# Mass-Weighted NEB Driver Script
# Generated by WKB_Tunneling_Analysis.generate_neb_script()

# Upload this file to the cluster alongside tunnel_code.py and any 
# input files, then submit through your scheduler:
#     python {script_name}
# """

# from tunkit import WKB_Tunneling_Analysis
# import sys

# wkb = WKB_Tunneling_Analysis()
# {coord_block}
# {load_block}
# {ts_block}
# # --- Set up ORCA header for NEB (engrad only, no freq) ---
# wkb.make_header_orca(
#     method='{method}',
#     basis_set='{basis_set}',
#     charge={charge},
#     multiplicity={multiplicity},
#     num_procs={num_procs},
#     mem='{mem}',
#     extra_options='{extra_options}',
#     make_header_neb=True,
# )

# {script_name_loader}

# # --- Run NEB ---
# wkb.run_neb(
#     reactant_coords=REACTANT_COORDS,
#     product_coords=PRODUCT_COORDS,
#     {ts_arg}
#     ts_mode={repr(ts_mode)},
#     n_images={n_images},
#     k_spring={k_spring},
#     max_steps={max_steps},
#     fmax_tol={fmax_tol},
#     frms_tol={frms_tol},
#     optimizer='{optimizer}',
#     scope='{scope}',
#     split_forces={split_forces},
#     precondition={precondition},
#     max_move={max_move},
#     lbfgs_mem={lbfgs_mem},
#     lbfgs_dr={lbfgs_dr},
#     validate_ts_gradient={validate_ts_gradient},
#     skip_alignment={skip_alignment},
#     align_every_step={align_every_step},
#     orca_path='{orca_path}',
#     n_parallel={n_parallel},
#     scratch_dir='{scratch_dir}',
#     job_prefix={job_prefix},
#     save_every={save_every},
# )

# print("\\nNEB finished. Download the scratch directory for results.")
# print("Next steps (run locally or in a new cluster job):")
# print("  1. make_header_orca(...) with freq+engrad")
# print("  2. make_projection_files_orca()")
# print("  3. Run ORCA frequency jobs")
# print("  4. extract_projection_energies_neb()")
# print("  5. build_reaction_path()")
# print("  6. calc_tunneling()")
# '''
#             with open(script_name, 'w') as f:
#                 f.write(script_content)
            
#             if verbose:
#                 print(f"Generated NEB driver script: {script_name}")
#                 print(f"Upload to cluster with tunnel_code.py and run.")








