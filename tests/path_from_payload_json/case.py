"""
Path construction from a payload JSON.

Covers:
  load_cluster_results   (also process_path_data)
  build_reaction_path    (also chains build_ds, sum_ds_to_path,
                          align_coordinates, mass_weight_coordinates)

Essential to all the rest of the code. Should be fixed first if any problems.
"""

# Parameters
json_file1 = 'tropolone_mwneb_projections.json'
invert_curve1 = False
json_file2 = 'carb_CH3_singlet_projections.json'
invert_curve2 = False

# Add tolerances when necessary. Everthing else is compared bitwise.
# TOLERANCES = {}    # curve_fit outputs may be sensitive to input last bit


def run(API):
    out = {}

    tun = API()
    tun.load_cluster_results(json_file=json_file1, invert_curve=invert_curve1, verbose=False)
    path = tun.build_reaction_path(verbose=False)

    tun2 = API()
    tun2.load_cluster_results(json_file=json_file2, invert_curve=invert_curve2, verbose=False)
    path2 = tun2.build_reaction_path(verbose=False)


    # from load_cluster_results
    out['energies']                = path.energies
    out['raw_coord']               = path.raw_coord
    out['raw_coord_ang']           = path.raw_coord_ang
    out['atomic_numbers']          = path.atomic_numbers
    out['masses']                  = path.masses
    out['proj_coordinates']        = path.proj_coordinates

    # from process_path_data
    out['proj_energies']           = path.proj_energies
    out['projected_corrections']   = path.projected_corrections
    out['projected_frequencies']   = path.projected_frequencies
    out['path_frequencies']        = path.path_frequencies
    out['gradient_rms']            = path.gradient_rms
    out['tangents']                = path.tangents

    # from build_reaction_path
    out['reaction_path']           = path.reaction_path
    out['mass_scaled_coord']       = path.mass_scaled_coord
    out['displacement_per_step']   = path.displacement_per_step
    out['displacement_non_mw']     = path.displacement_non_mw
    out['cartesian_disp_non_mw']   = path.cartesian_disp_non_mw

            # from load_cluster_results
    out['energies2']                = path2.energies
    out['raw_coord2']               = path2.raw_coord
    out['raw_coord_ang2']           = path2.raw_coord_ang
    out['atomic_numbers2']          = path2.atomic_numbers
    out['masses2']                  = path2.masses
    out['proj_coordinates2']        = path2.proj_coordinates

    # from process_path_data
    out['proj_energies2']           = path2.proj_energies
    out['projected_corrections2']   = path2.projected_corrections
    out['projected_frequencies2']   = path2.projected_frequencies
    out['path_frequencies2']        = path2.path_frequencies
    out['gradient_rms2']            = path2.gradient_rms
    out['tangents2']                = path2.tangents

    # from build_reaction_path
    out['reaction_path2']           = path2.reaction_path
    out['mass_scaled_coord2']       = path2.mass_scaled_coord
    out['displacement_per_step2']   = path2.displacement_per_step
    out['displacement_non_mw2']     = path2.displacement_non_mw
    out['cartesian_disp_non_mw2']   = path2.cartesian_disp_non_mw

#     c = tun.clip_reaction_path(clip_start=10, clip_end=5, plot_curve=False, verbose=False)
    clip_path = path.clipped(clip_start=10, clip_end=5)
    out['clip_reaction_path'] = clip_path.reaction_path
    out['clip_energies']      = clip_path.energies
    out['clip_proj_energies'] = clip_path.proj_energies
    out['clip_raw_coord_ang'] = clip_path.raw_coord_ang
    out['clip_n_images']      = len(clip_path.reaction_path)


    return out
