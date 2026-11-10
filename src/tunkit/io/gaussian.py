import numpy as np
import os
import re
from dataclasses import replace
from tunkit.core import elements as elem
from tunkit.core import constants as const
from tunkit.core import geometry as geom
from tunkit.path import projections
from tunkit.path import diagnostics









def extract_irc_gaussian(path,
                         irc_file_name, 
                         two_irc_files=False, 
                         first_irc_file_name='', 
                         second_irc_file_name='',
                         verbose=True):
    # if irc_file_name == '':
    #     irc_file_name = self.mep_file_name
    if not two_irc_files:
        if irc_file_name.strip() == '':
            print("No IRC file! Use the parameter 'irc_file_name' when calling the extract_irc method or when instanciating the Class.")
            raise ValueError("No IRC file provided")
        raw_coord_ang, raw_coord, irc_step, atomic_numbers = extract_coord(irc_file_name, verbose=verbose)
        energies = extract_energies(irc_file_name, verbose=verbose)
        atomic_numbers = elem.check_atomic_numbers(atomic_numbers)
        masses = elem.get_masses(atomic_numbers)
    else:
        if first_irc_file_name.strip() == '' or second_irc_file_name.strip() == '':
            print("One of the IRC files was not given! Use the parameters 'first_irc_file_name' and 'second_irc_file_name' when calling the extract_irc method.")
            raise ValueError("File not found!")
        raw_coord_ang, raw_coord, irc_step, atomic_numbers = extract_coord_2files(first_irc_file_name, second_irc_file_name, verbose=verbose)
        energies1 = extract_energies(first_irc_file_name, verbose=verbose)
        energies2 = extract_energies(second_irc_file_name, verbose=verbose)
        energies = np.array(energies1[::-1] + [0.0] + energies2)
        atomic_numbers = elem.check_atomic_numbers(atomic_numbers)
        masses = elem.get_masses(atomic_numbers)

    path = replace(path,
                   raw_coord_ang=raw_coord_ang, 
                   raw_coord=raw_coord, 
                   atomic_numbers=atomic_numbers, 
                   masses=masses, 
                   energies=energies
)

    return path







def extract_coord(mep_file_name, verbose=True):
    with open(mep_file_name) as fobj:
        path1 = []
        path2 = []
        an1 = []
        an2 = []
        dash_counter = 0
        rec = False
        an_point = []
        coord_point = []
        path = 1
        start = 0
        for line in fobj:
            if 'Step size                  ' in line:
                step = float(line.split()[3])
            if start == 0 and 'Point Number:   0' in line:
                start = 1
            if "Path Number:   2" in line:
                path = 2
            if "Input orientation" in line and start == 1:
                rec = True
                coord_point = []
                an_point = []
                continue
            if rec:
                if "------------------" in line:
                    dash_counter += 1
                    continue
            if dash_counter == 2:
                col = line.split()
                an_point.append(int(col[1]))
                coord_point.append([float(col[3]), 
                                    float(col[4]), 
                                    float(col[5])])
            if dash_counter == 3:
                dash_counter = 0
                rec = False
            if ' Path Number: ' in line  and coord_point != []:
                if path == 1:
                    path1.append(coord_point)
                    an1.append(an_point)
#                     coord_point = []
                elif path == 2:
                    path2.append(coord_point)
                    an2.append(an_point)
#                     coord_point = []
    if path == 2:
        zero = []
        for i in range(len(path1[0])):
            zero.append([(path1[0][i][0]+path2[0][i][0])/2,
                            (path1[0][i][1]+path2[0][i][1])/2,
                            (path1[0][i][2]+path2[0][i][2])/2])
        raw_coord_ang = np.array(path1[::-1] + [zero] + path2)
        raw_coord = raw_coord_ang * const.ang_to_bohr 
        irc_step  = step #[zero] +
        atomic_numbers = np.array(an1 + an2)
        if verbose:
            print('''Coordinates extracted from the IRC output file: {}'''.format(mep_file_name))
    else:
        print('Could only read the forward path from the IRC output file. Check the file '
              'and try again or use the xyz extract method (extract_coord_energy_xyz).')

    return raw_coord_ang, raw_coord, irc_step, atomic_numbers








def extract_energies(mep_file_name, verbose=True):
    energ1 = []
    energ2 = []
    path = 1
    rec= True
    # if file_name:
    #     mep_file_name = file_name
    # else:
    #     mep_file_name = self.mep_file_name
    with open(mep_file_name) as fobj:
        for line in fobj:
            if ' Path Number:   2' in line:
                path = 2
#             if True and 'Point Number:   0 ' in line: # (Inserir se necessário |R|)
#                 rec = True
#                 continue #(Inserir se necessário |R|)
            if 'SCF Done' in line:
                energy = line
            if rec:
                if ' Path Number: ' in line and path == 1:
                    # Get the energy in hartree
                    energ1.append(float(energy.split()[4]))
                elif ' Path Number: ' in line and path == 2:
                    # Get the energy in hartree
                    energ2.append(float(energy.split()[4]))
    if path == 1:
        energies = np.array(energ1)
    elif path == 2:
        energies = np.array(energ1[::-1] + energ2)
    if verbose:
        print('''Energies extracted from the IRC output file: {}'''.format(mep_file_name))
        bh = np.max(energies) - energies[0]
        print('EE barrier height is {:.2f} kcal/mol or {:.2f} cm-1.'.format(
                            bh * const.hartree_to_kcalmol, 
                            bh * const.hartree_to_cm)) 

    return energies








def extract_coord_2files(filename1, filename2, verbose=True):
    with open(filename1) as fobj:
        path1 = []
        an1_point = []
        an1 = []
        dash_counter = 0
        rec = False
        coord_point = []
        start = 0
        for line in fobj:
            if 'Step size                  ' in line:
                step = float(line.split()[3])
            if start == 0 and 'Point Number:   0' in line:
                start = 1
            if "Input orientation" in line and start == 1:
                rec = True
                coord_point = []
                an1_point = []
                continue
            if rec:
                if "------------------" in line:
                    dash_counter += 1
                    continue
            if dash_counter == 2:
                col = line.split()
                an1_point.append(int(col[1]))
                coord_point.append([float(col[3]), 
                                    float(col[4]), 
                                    float(col[5])])
            if dash_counter == 3:
                dash_counter = 0
                rec = False
            if ' Path Number: ' in line  and coord_point != []:
                path1.append(coord_point)
                an1.append(an1_point)
                
    with open(filename2) as fobj:
        path2 = []
        an2 = []
        an2_point = []
        dash_counter = 0
        rec = False
        coord_point = []
        start = 0
        for line in fobj:
            if 'Step size                  ' in line:
                step = float(line.split()[3])
            if start == 0 and 'Point Number:   0' in line:
                start = 1
            if "Input orientation" in line and start == 1:
                rec = True
                coord_point = []
                an2_point = []
                continue
            if rec:
                if "------------------" in line:
                    dash_counter += 1
                    continue
            if dash_counter == 2:
                col = line.split()
                an2_point.append(int(col[1]))
                coord_point.append([float(col[3]), 
                                    float(col[4]), 
                                    float(col[5])])
            if dash_counter == 3:
                dash_counter = 0
                rec = False
            if ' Path Number: ' in line  and coord_point != []:
                path2.append(coord_point)
                an2.append(an2_point)

    zero = []
    for i in range(len(path1[0])):
        zero.append([(path1[0][i][0]+path2[0][i][0])/2,
                            (path1[0][i][1]+path2[0][i][1])/2,
                            (path1[0][i][2]+path2[0][i][2])/2])
        
    raw_coord_ang = np.array(path1[::-1] + [zero] + path2)
    raw_coord = raw_coord_ang * const.ang_to_bohr
    irc_step  = step #[zero] +
    atomic_numbers = np.array(an1 + an2)
    if verbose:
        print('''Coordinates extracted from the IRC output files: {} and {}'''.format(filename1, filename2))

    return raw_coord_ang, raw_coord, irc_step, atomic_numbers








def extract_projection_energies(path, 
                                path_to_folder='', 
                                masses = [], 
                                diagnostic_path_freq=False, 
                                verbose=True, 
                                align=False, 
                                gnc_end=0.75):
    """
    Iterates through Gaussian .log files in the folder, extracts E, Grad, Hess,
    projects frequencies, and builds the ZPE-corrected energy (VAG).

    This method iterates through a folder containing Gaussian output files 
    (.log) for each point along a Nudged Elastic Band (NEB) or intrinsic reaction
    coordinates (IRC) path. For each image, it projects out the translation, rotation, 
    and gradient (reaction path) modes from the Hessian to obtain the true vibrational 
    frequencies orthogonal to the path.

    Parameters
    ----------
    path_to_folder : str, optional
        The directory path containing the .log files. If not provided,
        uses self.path_to_folder.
    verbose : bool, optional
        If True, prints progress and warning messages (e.g., missing file pairs). 
        Default is True.

    Attributes Updated
    ------------------
    self.masses : np.ndarray
        Array of atomic masses, in amu, (initialized from the first valid .log file).
    self.projected_corrections : np.ndarray
        Array of ZPE corrections (in Hartree) for each point on the path.
    self.proj_energies : np.ndarray
        Array of ZPE-corrected energies (Electronic Energy + ZPE) for each point.
    self.hessians : np.ndarray
        Array of full Hessian matrices for each point.
    self.tangents : np.ndarray
        Array of gradient vectors for each point.
    self.proj_coordinates : np.ndarray
        Array of geometries (coordinates) for each point.
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
    # if path_to_folder:
    #     self.path_to_folder = path_to_folder
    if not path_to_folder:
        raise ValueError('The path to the projections folder must be entered!')


    # SET MASSES BEFORE EXTRACTIONS FOR MASS WEIGHTED PROJECTIONS!!!!
    # if masses:
    #     self.masses = np.array(masses) 

    # Get list of .log files and sort them (assuming names like 001.log, 002.log)
    filenames = [f for f in os.listdir(path_to_folder) if f.endswith('.log') or f.endswith('.out')]
    filenames.sort()
    
    if not filenames:
        print(f"No .log or .out files found in {path_to_folder}")
        return

    n_files = len(filenames)

    # Initialize storage
    energies = []
    tangents = []
    hessians = []
    mw_hessians = []
    mw_gradients = []
    coords_list = []
    corrections = []
    path_freqs = [] # To store the unprojected frequencies
    proj_freqs_list = []
    gradient_rms = []

    if verbose:
        print(f"Processing {len(filenames)} Gaussian output files in {path_to_folder}...")
        if len(masses) != 0:
            print('Custom masses provided, so they are going to be used.')

    for n, filename in enumerate(filenames):
        filepath = os.path.join(path_to_folder, filename)

        try:
            # These coordinates are typically Standard Orientation (Angstroms)
            coords, hess, grad, energy, atomic_numbers = parse_gaussian_archive(filepath)

            # Initialize atomic info if missing (e.g. if we didn't load an MEP file first)
            # if len(self.atomic_numbers) == 0:
            #     self.atomic_numbers = np.array(atomic_numbers)

            if len(masses) == 0:
                masses = elem.get_masses(atomic_numbers) # Uses atomic_numbers to fill masses

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            # We should probably stop or skip, but skipping breaks the path index. 
            # It's better to stop.
            return

        energies.append(energy)
        tangents.append(grad)
        hessians.append(hess)
        coords_list.append(coords)
        
    if align:
        center_of_mass, rotational_matrices, transl_vectors, alignement_rmsd, aligned_coord = geom.align_coordinates(coords_list, masses)
        coords_list = aligned_coord
        path = replace(path,                       
                       center_of_mass=center_of_mass, 
                       rotational_matrices=rotational_matrices, 
                       transl_vectors=transl_vectors, 
                       alignement_rmsd=alignement_rmsd)
    
    # Tangents via central differences
    # This is smoother than the 'current - neighbor' logic
    # np.gradient uses (x[i+1] - x[i-1])/2 for interior points
    tangents = np.gradient(coords_list, axis=0) 


    for n in range(n_files):
        # We NEED flat tangents, because we are going to add them to the D matrix 
        vec_tangent = tangents[n].flatten()
        proj_freqs, hess_mw, grad_mw = projections.project_frequencies(
            masses, coords_list[n], hessians[n], vec_tangent, threshold=1.0E-4
        )
        
        proj_freqs_list.append(proj_freqs)
        mw_hessians.append(hess_mw)
        mw_gradients.append(grad_mw)

        current_rms = np.sqrt(np.mean(grad_mw**2))
        gradient_rms.append(current_rms)
        
        # cutoff=10.0 ignores immaginary modes smaller than 10,0g
        # [TODO] there is the issue of how to deal with the imaginary frequencies
        zpe_hartree = projections.calculate_zpe(proj_freqs, cutoff=10.0)
        corrections.append(zpe_hartree)


    proj_energies = np.array(energies) + np.array(corrections)
    energies = np.array(energies) # Store raw electronic energies
    projected_corrections = np.array(corrections)

    # self.grandients = np.array(gradients) We are using the tangent as the gradient
    gradients = np.array(tangents)
    mw_gradients = np.array(mw_gradients)
    
    hessians = np.array(hessians)
    mw_hessians = np.array(mw_hessians)
    
    proj_coordinates = np.array(coords_list)
    raw_coord = np.array(coords_list) # Update raw_coord for other methods (alignment etc)
    
    # self.path_frequencies = np.array(path_freqs, dtype=object)
    
    if verbose:
        print(f'Projected energies have been extracted from folder {path_to_folder}.')

    
    # Simple diagonalization of MW Hessian to get normal modes (Mass-weighted eigenvectors)
    # Can't be done in the project_frequencies method because it's matrix also projects
    # [TODO] should probably be made into its own method to be reused
    all_modes = []
    for n, h_mw in enumerate(mw_hessians):
        clean_h_mw = projections._project_out_tr(raw_coord[n], h_mw, masses)
        vals, vecs = np.linalg.eigh(clean_h_mw)
        all_modes.append(vecs) # Shape (3N, 3N)
        
        # Convert eigenvalue to frequency (cm^-1)
        path_freqs.append(geom.sqrt_eigval_convert_to_cm(vals))
    
    path_frequencies = np.array(path_freqs)
    normal_modes = np.array(all_modes)


    if diagnostic_path_freq:
        diagnostics.diagnostic_path_and_proj_freq(n_files, proj_freqs_list, path_frequencies)


    # barrier heitght
    if verbose:
        bh = np.max(proj_energies) - proj_energies[0]
        print('V_a barrier height is {:.2f} kcal/mol or {:.2f} cm-1.'.format(
                                bh * const.hartree_to_kcalmol, 
                                bh * const.hartree_to_cm))

    return replace(path,
                   proj_energies=proj_energies, 
                   energies=energies, 
                   projected_corrections=projected_corrections, 
                   tangents=tangents, 
                   gradients=gradients,
                   mw_gradients=mw_gradients, 
                   hessians=hessians,
                   mw_hessians=mw_hessians,
                   proj_coordinates=proj_coordinates,
                   raw_coord=raw_coord,
                   path_frequencies=path_frequencies,
                   normal_modes=normal_modes,
                   masses=masses,
                   atomic_numbers=atomic_numbers)













def _parse_gaussian_log_frequencies(filename):
    """
    Parses the standard harmonic frequencies from a Gaussian log file.
    Returns a list of floats (cm^-1).
    """
    freqs = []
    with open(filename, 'r') as f:
        for line in f:
            if "Frequencies --" in line:
                # Example line: " Frequencies --    127.1034   135.7404   215.8281"
                parts = line.split('--')[1].split()
                freqs.extend([float(x) for x in parts])
    return np.array(freqs)








def parse_gaussian_archive(filename):
    """
    Parses Gaussian 16 Archive blocks to extract Geometry, Hessian, Gradient, and Energy.
    
    Returns:
    --------
    coords_ang : np.array (N_atoms, 3)
    hessian : np.array (3N, 3N) in Hartree/Bohr^2
    gradient : np.array (3N) in Hartree/Bohr
    energy : float (Hartree)
    atomic_numbers : list (integers)
    """
    with open(filename, 'r') as f:
        content = f.read()


    # what a nightmare... greed match for the block
    match = re.search(r'1\\1\\GINC.*?(?=\s*@)', content, re.DOTALL)
    if not match:
        match = re.search(r'1\\1\\.*?(?=\s*@)', content, re.DOTALL)
        
    if not match:
        # sometimes the file ends abruptly, try to get what's there (should we?)
        match = re.search(r'1\\1\\.*', content, re.DOTALL)
        if not match:
            raise ValueError(f"Could not find Gaussian Archive block in {filename}.")
    
    # flatten to big string
    archive_str = match.group(0).replace('\n', '').replace(' ', '')

    # search for pattern like "\\HF=-123.456\\" or "\\MP2=-123.456\\"
    # look for a number preceded by "=" and bounded by backslashes
    energy = None
    # common Gaussian Energy tags (what a nightmare... cclib ???) 
    tags = ['CCSD(T)', 'CCSD', 'MP2', 'HF', 'E'] 
    
    sections = archive_str.split('\\')
    for section in sections:
        for tag in tags:
            if section.startswith(f"{tag}="):
                try:
                    # Extract the number part
                    val_str = section.split('=')[1]
                    energy = float(val_str)
                    break
                except:
                    continue
        if energy is not None:
            break
    
    if energy is None:
        # fallback for excited states 
        # look for the State=X-A1g|E=-123.123 format 
        for section in sections:
            if "State=" in section and "E=" in section:
                try:
                    parts = section.split('|')
                    for p in parts:
                        if p.startswith("E="):
                            energy = float(p.split('=')[1])
                except:
                    pass
    
    if energy is None:
        raise ValueError(f"Could not extract Energy from archive in {filename}")


    # string sections are separated by \\
    # assume the gradient and hessian blocks are the longest comma-separated sequences
    
    # find the geometry section first to known N atoms
    # geometry usually follows the Charge,Multiplicity section
    idx = 0
    start_geom = -1
    
    # charge,multiplicity (two integers)
    for i, token in enumerate(sections):
        if ',' in token:
            parts = token.split(',')
            if len(parts) == 2 and parts[0].lstrip('-').isdigit() and parts[1].isdigit():
                start_geom = i + 1
                break
    
    if start_geom == -1:
            raise ValueError("Could not find Charge/Multiplicity section.")

    atoms = []
    atomic_numbers = []
    coords_ang = []
    
    # parse geometry
    # tt ends when we hit a token that contains "Version=" or other metadata, or is empty
    curr = start_geom
    while curr < len(sections):
        token = sections[curr]
        if not token or "Version=" in token or "State=" in token or "HF=" in token: 
            break
        
        # format: label,X,Y,Z
        atom_data = token.split(',')
        if len(atom_data) >= 4:
            atoms.append(atom_data[0])
            # Angstroms
            coords_ang.append([float(x) for x in atom_data[1:4]])
            
            # get atomic numbers
            an = elem.atomic_number_from_symbol(atom_data[0])
            if an:
                atomic_numbers.append(an)
            else:
                # if it is already an int just get it (what a nightmare)
                if atom_data[0].isdigit():
                    atomic_numbers.append(int(atom_data[0]))
                else:
                    print(f"Warning: Unknown atom symbol {atom_data[0]}")
                    atomic_numbers.append(0)

        curr += 1
        
    coords_ang = np.array(coords_ang)
    
    # sanitize to default units
    coords_bohr = coords_ang * const.ang_to_bohr

    # filter sections that look like arrays of floats.
    numeric_blocks = []
    
    for block in sections:

        if block.count(',') > 5:
            # filters letters except E 
            if re.search(r'[A-DF-Za-df-z]', block): 
                continue
            
            # check if purely numeric
            try:
                # clean prefixe like "NImag=1\\"
                clean_b = block.split('\\')[-1] 
                vals = [float(x) for x in clean_b.split(',') if x]
                numeric_blocks.append(vals)
            except:
                pass
    
    # sort by length (hessian larger 3N*3N)
    numeric_blocks.sort(key=len, reverse=True)
    
    if len(numeric_blocks) < 2:
        raise ValueError(f"Could not locate both Hessian and Gradient in {filename}")

    hess_vals = numeric_blocks[0] # largest
    grad_vals = numeric_blocks[1] 
    

    gradient = np.array(grad_vals)
    
    # reconstruct hessian because lower triangluar packed
    dim = 3 * len(atoms)
    hessian = np.zeros((dim, dim))
    k = 0
    for i in range(dim):
        for j in range(i + 1):
            if k < len(hess_vals):
                val = hess_vals[k]
                hessian[i, j] = val
                hessian[j, i] = val
                k += 1
    
    return coords_bohr, hessian, gradient, energy, atomic_numbers
    