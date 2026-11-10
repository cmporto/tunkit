import os
from tunkit.core import elements as elem








def make_projection_files(mep_file_name, coords, atomic_numbers, header=''):

    split_path = os.path.split(mep_file_name)
    path = os.path.join(*split_path[:-1])
    directory = os.path.join(path, 'projections_{}'.format(split_path[-1].split('.')[0]))
    # self.path_to_folder = directory
    if not os. path. exists(directory): 
        os.mkdir(os.path.join(directory))
    if not mep_file_name.endswith('xyz'): # All gaussian files are valid, except xyz... [TODO] Maybe do a case
        header = extract_header(mep_file_name)
    elif not header:
        raise ValueError("Could not extract the header from the file or using an xyz file. "
                         "In this case, supply the header made with ee_inputs.make_header().")
    for num, coordinate in enumerate(coords):
        # We assume that we're not going to do more than 999 frequencies...
        projection_file = '{:03d}.com'.format(num)
        with open(os.path.join(directory, projection_file), 'w') as fo:
            fo.write(header.replace('subchksub', projection_file.split('.')[0]))
            for n, atom in enumerate(coordinate):
                fo.write(' {} {:15.6f} {:15.6f} {:15.6f}\n'.format(elem.elements[atomic_numbers[n]][0], 
                                                                    float(atom[0]) ,
                                                                    float(atom[1]) ,
                                                                    float(atom[2]) ))
            fo.write('\n\n')

    print("Finished projections!")








def extract_header(file_name, verbose=True):
    g_options = ''
    hash_options = ''
    rec = False
    dash = 0
    with open(file_name) as fo:
        count_stars = 0
        for line in fo:
#             print(dash, rec, count_stars)
            if 'Multiplicity' in line:
                s = line.split()
                charge = s[2]
                multi = s[5]
            if '******************************************' in line:
                count_stars += 1
                continue
            if count_stars == 2:
                rec = True
            if '----------------------------' in line and count_stars == 2:
                dash += 1
                continue
            if dash == 1:
                hash_options += line.strip()
                continue
            if dash == 2:
                rec = False
                dash += 1    # increase dash to stop reccing
                count_stars = 3
            if rec:
                if '%chk' in line:
#                     s = line.split('=')
                    line = '%chk=subchksub.chk'
                g_options += line.strip() + '\n'
    
    options = ''
    for item in hash_options.split():
        if 'irc' in item:
            continue
        options += item + ' '
    hash_options = options    
    
    raw_header = '{}{} freq=(noraman,tprojected)\n\nProjected path for {}\n\n{} {}\n'.format(g_options, 
                                                                                                hash_options,
                                                                                                file_name, 
                                                                                                charge, 
                                                                                                multi)
    header = ''
    for line in raw_header.split('\n'):
        if 'Will use up' not in line:
            header += line + '\n'
    return header.strip() + '\n'








def make_header(mep_file_name,
                method='',
                basis_set='',
                charge=0,
                multiplicity=1, 
                num_procs=1,
                mem='1GB',
                extra_options='',
                verbose=True):
    g_options = f'''%nprocshared={num_procs}
%mem={mem}
%chk=subchksub.chk\n'''
    hash_options = f'{method}/{basis_set} '       
    raw_header = '{}# {} {} freq=(noraman,tprojected)\n\nProjected path for {}\n\n{} {}\n'.format(g_options, 
                                                                                                    hash_options, 
                                                                                                    extra_options, 
                                                                                                    mep_file_name, 
                                                                                                    charge, 
                                                                                                    multiplicity)
    return raw_header.strip() + '\n'








def make_projection_files_orca(coord_ang, atomic_numbers, header='', path_to_folder='projections_scratch', verbose=False):
    if not header:
        raise ValueError('Header must be input as parameter.')
    
    if not os.path.exists(path_to_folder): 
        os.makedirs(path_to_folder, exist_ok=True)

    for num, coordinate in enumerate(coord_ang):
        projection_file = '{:03d}.inp'.format(num)
        with open(os.path.join(path_to_folder, projection_file), 'w') as fo:
            fo.write(header)
            for n, atom in enumerate(coordinate):
                fo.write(' {} {:15.6f} {:15.6f} {:15.6f}\n'.format(elem.elements[atomic_numbers[n]][0], 
                                                                    atom[0],
                                                                    atom[1],
                                                                    atom[2]))
            fo.write('*')

    if verbose:
        print("Finished projections!")

    return path_to_folder








def make_header_orca(method='',
                basis_set='',
                charge=0,
                multiplicity=1, 
                num_procs=1,
                mem='1000',
                extra_options='',
                verbose=True,
                make_header_neb=False):
    if not make_header_neb:
        header = f'''  !freq engrad tightscf 
!{method} {basis_set}
{extra_options}
%maxcore {mem}
%pal nprocs {num_procs} end
* xyz {charge} {multiplicity}
'''
        return header
    else: # neb_header has no freqs in it! Also cleaner
        neb_header = f'''  ! engrad tightscf 
!{method} {basis_set}
{extra_options}
%maxcore {mem}
%pal nprocs {num_procs} end
* xyz {charge} {multiplicity}
'''
        return neb_header
        









# class EEInputsIOMixin:






































