import numpy as np
import os
from dataclasses import replace
from tunkit.core import constants as const
from tunkit.core import elements as elem
from tunkit.core import geometry as geom
from tunkit.path import projections 
from tunkit.path import diagnostics








def extract_coord_energy_xyz(path, mep_file_name, verbose=True):
    
    with open(mep_file_name) as fo:
        check_n_atoms = True
        coords = []
        coord = []
        an = []
        at_numbers = []
        energies = []
        count = 0
        for line in fo:
            count += 1
            if check_n_atoms:
                n_atoms = int(line.strip())
                check_n_atoms = False
            if count <= n_atoms + 2:
                pass
            else:
                count = 1
                at_numbers.append(an)
                coords.append(coord)
                coord = []
                an = []
                continue
            if count == 2:
                energies.append(float(line.split()[-1]))
                continue
            if count > 2:
                s = line.split()
                an.append(elem.search_element(s[0]))
                coord.append([float(s[1]), float(s[2]), float(s[3])])
        coords.append(coord)
        at_numbers.append(an)
    atomic_numbers = np.array(at_numbers)
    raw_coord_ang = np.array(coords)
    raw_coord = raw_coord_ang * const.ang_to_bohr
    energies  = np.array(energies)
    atomic_numbers = elem.check_atomic_numbers(atomic_numbers)
    masses = elem.get_masses(atomic_numbers)

    if verbose:
        print('''Coordinates and energies extracted from the xyz output file: {}'''.format(mep_file_name))
        bh = np.max(energies) - energies[0]
        print('EE barrier height is {:.2f} kcal/mol or {:.2f} cm-1.'.format(
                            bh * const.hartree_to_kcalmol, 
                            bh * const.hartree_to_cm)) 

    path = replace(path,
                   atomic_numbers=atomic_numbers, 
                   raw_coord_ang=raw_coord_ang, 
                   raw_coord=raw_coord, 
                   energies=energies, 
                   masses=masses)

    return path









def extract_projection_energies_neb(path, 
                                    path_to_folder='', 
                                    invert_curve=False,
                                    threshold=1.0E-4,
                                    masses=None,
                                    diagnostic_path_freq=False,
                                    verbose=True):
    """
    Extracts electronic energies, gradients, and Hessians from a folder of NEB images, 
    computes projected vibrational frequencies, and calculates the Zero Point Energy (ZPE) 
    corrected path.

    This method iterates through a folder containing matched pairs of ORCA output files 
    (.engrad and .hess) for each point along a Nudged Elastic Band (NEB) path. For each 
    image, it projects out the translation, rotation, and gradient (reaction path) modes 
    from the Hessian to obtain the vibrational frequencies orthogonal to the path.

    Parameters
    ----------
    path_to_folder : str, optional
        The directory path containing the .engrad and .hess files. If not provided,
        uses self.path_to_folder.
    verbose : bool, optional
        If True, prints progress and warning messages (e.g., missing file pairs). 
        Default is True.

    Attributes Updated
    ------------------
    self.masses : np.ndarray
        Array of atomic masses, amu (initialized from the first valid .hess file).
    self.projected_corrections : np.ndarray
        Array of ZPE corrections (in Hartree) for each point on the path.
    self.proj_energies : np.ndarray
        Array of ZPE-corrected energies (Electronic Energy + ZPE) for each point.
    self.hessians : np.ndarray
        Array of full Hessian matrices for each point.
    self.gradients : np.ndarray
        Array of gradient vectors for each point.
    self.proj_coordinates : np.ndarray
        Array of geometries (coordinates) for eahc point.
    self.normal_modes : np.ndarray
        Array containing normal mode data read from files.
    self.energies : np.ndarray
        Array of raw electronic energies (without ZPE) if not already set.

    Returns
    -------
    None
        The method updates class attributes in-place and prints the calculated 
        barrier height.
    """

    # SET MASSES BEFORE EXTRACTIONS FOR MASS WEIGHTED PROJECTIONS!!!!
    if masses is not None:
        if verbose:
            print('Custom masses added!')
        masses = np.array(masses)
    else:
        masses = np.array([])


    # if path_to_folder:
    #     self.path_to_folder = path_to_folder
    # elif not self.path_to_folder:
    #     print('The path to the projections folder must be entered!')
    #     return # Stop execution if no path

    all_files = sorted(os.listdir(path_to_folder))
    
    # creates a set of unique names to then read them
    stems = sorted(list(set([os.path.splitext(f)[0] for f in all_files if f.endswith('.engrad') or f.endswith('.hess')])),
                    reverse=invert_curve) # If we need to invert the curve, this is where it does!!

    energies = []
    correction = []
    gradients = []
    gradient_rms = []
    hessian_list = [] 
    mw_hessian_list = []
    mw_gradient_list = []
    proj_coordinates = []
    normal_mod = []
    atomic_numbers = []
    path_freqs = []
    proj_freqs_list = []
    flat_tangents = []

    if verbose:
        print(f"Processing {len(stems)} points along the path...")

    # couting the number of valid files, because there may be different files in the folder
    n_steps = 0

    for n, stem in enumerate(stems):
        engrad_path = os.path.join(path_to_folder, stem + '.engrad')
        hess_path = os.path.join(path_to_folder, stem + '.hess')

        # check if both files exist
        if not os.path.exists(engrad_path) or not os.path.exists(hess_path):
            if verbose: print(f"Skipping {stem}: Missing .engrad or .hess file pair. Go check!")
            continue

        n_steps += 1


        gradient, energy = orca_engrad_parser(engrad_path)
        

        masses_read, atoms, coord, hess, read_n_modes, freqs = read_orca_data(hess_path)


        energies.append(energy)
        gradients.append(gradient)
        hessian_list.append(hess)
        proj_coordinates.append(coord)

        # populate the atomic labels
        if not atomic_numbers: 
            for atom in atoms:
                atomic_numbers.append(elem.atomic_number_from_symbol(atom))

        # populate masses if the user ddin't provide one 
        if len(masses) == 0: 
            if verbose:
                print('Using default masses!')
            masses = np.array(masses_read) 

    # [FIXED] The extracted coordinates were not aligned before calculating the tangent.
    center_of_mass, rotational_matrices, transl_vectors, \
        alignement_rmsd, aligned_coord = geom.align_coordinates(proj_coordinates, masses)
    proj_coordinates = list(aligned_coord)


    # calculate tangents central differences
    # This is smoother than the previous 'current - neighbor' logic
    # np.gradient uses (x[i+1] - x[i-1])/2 for interior points
    tangents = np.gradient(proj_coordinates, axis=0)



    for n in range(n_steps):
        # we need to flatten the tangents to go into the D matrix
        tangent = tangents[n].flatten()
        flat_tangents.append(tangent)

        proj_freqs, hessian_mw, gradient_mw = projections.project_frequencies(masses, 
                                                                        proj_coordinates[n], 
                                                                        hessian_list[n], 
                                                                        tangent, 
                                                                        threshold=threshold)
        
        proj_freqs_list.append(proj_freqs)
        mw_hessian_list.append(hessian_mw)
        mw_gradient_list.append(gradient_mw)


        current_rms = np.sqrt(np.mean(gradient_mw**2))
        gradient_rms.append(current_rms)


        zpe = projections.calculate_zpe(proj_freqs)
        correction.append(zpe)
        
        # [FIXED] Orca's frequencies are pre-weighted so the isotopic substitution 
        # was not working.
        clean_h_mw = projections._project_out_tr(proj_coordinates[n], hessian_mw, masses)
        vals, vecs = np.linalg.eigh(clean_h_mw)
        

        # convert eigenvalues to frequencies in cm^-1
        path_freqs.append(geom.sqrt_eigval_convert_to_cm(vals))

        normal_mod.append(vecs)


    projected_corrections = np.array(correction)
    hessians = np.array(hessian_list)
    mw_hessians = np.array(mw_hessian_list)
    proj_coordinates = np.array(proj_coordinates)
    normal_modes = np.array(normal_mod)
    # self.gradients = np.array(gradients) # The gradient we are using are the tangens
    gradients = np.array(tangents)
    mw_gradients = np.array(mw_gradient_list)
    atomic_numbers = np.array(atomic_numbers)
    path_frequencies = np.array(path_freqs)
    tangents = np.array(tangents)
    projected_frequencies = np.array(proj_freqs_list)
    gradient_rms=np.array(gradient_rms)
    
    # populating the raw_coords if empty (bohrs). [TODO] TOO MANY COORDS. We need to choose one, or probably regret in the future...
    # if len(raw_coord) == 0: 
    raw_coord = proj_coordinates

    # making the adiabatic MEP with the ZPE
    proj_energies = np.array(energies) + projected_corrections
    
    # [TODO] TOO MANY ENERGIES. Same with the coords. Choose one and stick to it... (maybe not... having all of them available is useful...)
    # if len(energies) == 0:
    energies = np.array(energies)

    if verbose:
        print(f'Projected energies have been extracted from folder {path_to_folder}.')

    if diagnostic_path_freq:
        diagnostics.diagnostic_path_and_proj_freq(n_steps, proj_freqs_list, path_frequencies)

    if len(proj_energies) > 0:
        # print(self.proj_energies)
        bh = np.max(proj_energies) - proj_energies[0]
        # print(bh)

        kcal_val = bh * const.hartree_to_kcalmol
        cm_val = bh * const.hartree_to_cm

        if verbose:
            print('Vª barrier height is {:.2f} kcal/mol or {:.2f} cm-1.'.format(kcal_val, cm_val))


    return replace(path,
                   center_of_mass=center_of_mass, 
                   rotational_matrices=rotational_matrices, 
                   transl_vectors=transl_vectors,
                   alignement_rmsd=alignement_rmsd,
                   projected_corrections=projected_corrections,
                   hessians=hessians,
                   mw_hessians=mw_hessians,
                   proj_coordinates=proj_coordinates,
                   normal_modes=normal_modes,
                   gradients=gradients,
                   mw_gradients=mw_gradients,
                   atomic_numbers=atomic_numbers,
                   path_frequencies=path_frequencies,
                   tangents=tangents,
                   projected_frequencies=projected_frequencies,
                   raw_coord=raw_coord,
                   proj_energies=proj_energies,
                   energies=energies,
                   gradient_rms=gradient_rms,
                   masses=masses)









def orca_engrad_parser(filename):
    gradient = []
    energy = None
    with open(filename) as fo:
        # There are the number of atoms, the energy, and then the gradient
        nblocks = 0
        for line in fo:
            # print(line.strip(), nblocks)
            if '#' not in line:
                nblocks += 1
            if nblocks >= 3 and '#' in line:
                    break
            if nblocks >= 3:
                gradient.append(float(line.strip()))
                continue
            # print(nblocks)
            if nblocks == 2 and '#' not in line:
                energy = float(line.strip())
    return gradient, energy








def read_orca_data(filename):
    """
    Orca .hess file parser for Geometry, Masses, Hessian, and Normal Modes.
    Separates blocks to be flexible to block order and '#' comments.
    Coordinates are also found in the .engrad file, but the default is to
    extract and use the ones here (.hess).
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    atoms = []
    masses = []
    coords = []
    hessian = None
    normal_modes = None
    freqs = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Lines starting with # are considered comments and skipped
        if not line or line.startswith('#'):
            i += 1
            continue

        # Parsing atom labels, masses and coordinates (bohr)
        if line.startswith("$atoms"):
            i += 1
            # Skipping number of atoms, but it could be stored from here
            # and used throughout the code...
            if i < len(lines) and lines[i].strip().isdigit(): 
                i += 1 
            
            while i < len(lines):
                curr = lines[i].strip()
                if curr.startswith("$"): # Start of a new block
                    break 
                if curr.startswith("#"): # Skip possible comments ()
                    i += 1
                    continue
                
                parts = curr.split()
                if len(parts) >= 5:
                    atoms.append(parts[0])
                    # The atoms: label  mass x y z (in bohrs)
                    masses.append(float(parts[1]))
                    coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
                i += 1
            continue 

        # Hessian blcok
        elif line.startswith("$hessian"):
            i += 1
            if i < len(lines):
                # Assume first line is the dimension... 
                # We build the hessian dimensions from it now (avoid mismatch later...)
                dim = int(lines[i].strip())
                hessian = np.zeros((dim, dim))
                i += 1 
                
                current_cols = []
                while i < len(lines):
                    curr = lines[i].strip()
                    if curr.startswith("$"): # Start of a new block
                        break 
                    if not curr or curr.startswith("#"): # Skip empty lines and possible comments
                        i += 1
                        continue
                    
                    parts = curr.split()
                    
                    # Distinguish Header vs Data
                    # Data lines have float in 2nd position (e.g. "0   0.123 ...")
                    is_data_line = False
                    if len(parts) > 1:
                        if "." in parts[1] or "E" in parts[1]: # [TODO] Use regex maybe...?
                            is_data_line = True
                    
                    if not is_data_line:
                        # Header: "    0    1    2    3"
                        current_cols = [int(c) for c in parts if c.lstrip('-').isdigit()]
                    else:
                        # Data: "0   0.123   0.456"
                        row_idx = int(parts[0])
                        vals = [float(x) for x in parts[1:]]
                        for k, val in enumerate(vals):
                            if k < len(current_cols): # Do I really need to check? Is this too much?
                                hessian[row_idx, current_cols[k]] = val
                    i += 1
            continue

        # Normal modes block. Normal modes are not mass weighted, so we have to diagonalize
        # the hessian again to do isotopic substitution...
        elif line.startswith("$normal_modes"):
            i += 1
            if i < len(lines):
                dims = lines[i].split()
                dim_r, dim_c = int(dims[0]), int(dims[1])
                normal_modes = np.zeros((dim_r, dim_c))
                i += 1 
                
                current_cols = []
                while i < len(lines):
                    curr = lines[i].strip()
                    if curr.startswith("$"): 
                        break
                    if not curr or curr.startswith("#"):
                        i += 1
                        continue
                    
                    parts = curr.split()
                    
                    is_data_line = False
                    if len(parts) > 1:
                        if "." in parts[1] or "E" in parts[1]:
                            is_data_line = True
                            
                    if not is_data_line:
                        current_cols = [int(c) for c in parts if c.lstrip('-').isdigit()]
                    else:
                        row_idx = int(parts[0])
                        vals = [float(x) for x in parts[1:]]
                        for k, val in enumerate(vals):
                            if k < len(current_cols):
                                normal_modes[row_idx, current_cols[k]] = val
                    i += 1
            continue
        
        # Not projected!! Can be used only for comparison...
        elif line.startswith("$vibrational_frequencies"):
            i += 2  # +2 to skip number of modes (3N)
            while i < len(lines):
                if not lines[i].strip() or lines[i].strip().startswith("$"): # Skip empty lines and stop at teh start of next block
                    break
                freqs.append(float(lines[i].split()[1]))
                i += 1
        i += 1

    return np.array(masses), atoms, np.array(coords), np.array(hessian), np.array(normal_modes), np.array(freqs)








def write_engrad_inputs(images_cart,
                        work_dir,
                        header,
                        atomic_numbers,
                        indices=None):
    """
    Write ORCA energy + gradient inputs, image_XXX.inp, one per geometry.

    The header comes from ee_inputs.make_header_orca(make_header_neb=True).
    Coordinates are written in Angstrom (ORCA's default for xyz input).
    
    Parameters
    ----------
    images_cart : list of ndarray, each (N_atoms, 3)
        Image coordinates in bohr.
    work_dir : str
        Directory for input/output files. Created if it doesn't exist.
    indices : list of int, optional
        Which image indices to write. If None, writes all interior 
        images (1 to N-2). Use [0, N-1] for endpoints. (Future implementation)
        
    Returns
    -------
    inp_files : list of (int, str)
        Pairs of (image_index, input_file_path).
    """
    os.makedirs(work_dir, exist_ok=True)
    
    if indices is None:
        indices = list(range(1, len(images_cart) - 1))
    
    inp_files = []
    for i in indices:
        inp_path = os.path.join(work_dir, f"image_{i:03d}.inp")

        with open(inp_path, 'w') as f:
            f.write(engrad_input_text(images_cart[i], header, atomic_numbers))

        inp_files.append((i, inp_path))

    return inp_files







def engrad_input_text(coords_cart, header, atomic_numbers):
    """
    Text of one ORCA engrad input: header + xyz block in Angstrom.
    read_input_xyz reads it back.
    """
    coords_ang = coords_cart * const.bohr_to_ang
    lines = [header]
    for n, atom in enumerate(coords_ang):
        sym = elem.elements[atomic_numbers[n]][0]
        lines.append(f' {sym:2s} {atom[0]:16.10f} {atom[1]:16.10f} {atom[2]:16.10f}\n')
    lines.append('*\n')
    return ''.join(lines)







def read_input_xyz(inp_path):
    """
    Read the xyz block of an ORCA input (the inverse of engrad_input_text).

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