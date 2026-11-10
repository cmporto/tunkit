import numpy as np
from tunkit.core import elements as elem
from tunkit.core import geometry as geom
from tunkit.plotting import path as plotpath






def _calc_arc_lengths(coords):
    """Calculates Euclidean distances between adjacent images in a trajectory."""
    lengths = []
    for i in range(len(coords) - 1):
        lengths.append(np.linalg.norm(coords[i+1].flatten() - coords[i].flatten()))
    return np.array(lengths)








def _calc_kinking_angles(coords):
    """Calculates the cosine of the angle between adjacent path segments."""
    angles = []
    for i in range(1, len(coords) - 1):
        v1 = coords[i].flatten() - coords[i-1].flatten()
        v2 = coords[i+1].flatten() - coords[i].flatten()
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-12 and n2 > 1e-12:
            cos_theta = np.dot(v1, v2) / (n1 * n2)
            angles.append(cos_theta)
        else:
            angles.append(1.0) # Assumes no kinking if no movement
    return np.array(angles)








def _calc_tangent_consistency(coords):
    """Calculates the change in normalized geometric tangent vectors between adjacent images."""
    # Calculate raw geometric tangents via central difference
    raw_tangents = np.gradient(coords, axis=0)
    deltas = []
    for i in range(1, len(raw_tangents)):
        t1 = raw_tangents[i-1].flatten()
        t2 = raw_tangents[i].flatten()
        
        # Normalize to compare pure directional changes
        n1, n2 = np.linalg.norm(t1), np.linalg.norm(t2)
        t1 = t1 / n1 if n1 > 1e-14 else t1
        t2 = t2 / n2 if n2 > 1e-14 else t2
        
        deltas.append(np.linalg.norm(t2 - t1))
    return np.array(deltas)








def _calc_gradient_overlaps(grads):
    """Calculates the normalized dot product of adjacent gradient vectors."""
    overlaps = []
    for i in range(len(grads) - 1):
        g1 = grads[i].flatten()
        g2 = grads[i+1].flatten()
        n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
        if n1 > 1e-12 and n2 > 1e-12:
            overlaps.append(np.dot(g1, g2) / (n1 * n2))
        else:
            overlaps.append(0.0)
    return np.array(overlaps)








def diagnostic_path_and_proj_freq(n_steps, all_proj_freqs, path_frequencies, noise_floor = -1.0):

    print("\n" + "="*60)
    print("PATH DIAGNOSTICS: Imaginary Frequency Check")
    print("="*60)
    
    # The noise floor should probably be allowed to be set by the user
    noise_floor = -1.0 # Initially I don't want to ignore anything. Maybe for production code.
    issue_found = False

    for n in range(n_steps):
        # Ideally a path should not have more than 1 imaginary frequency 
        raw_imags = [f for f in path_frequencies[n] if f < noise_floor]
        
        if len(raw_imags) > 1:
            issue_found = True
            # max_raw = min(raw_imags) # min() gets the most negative value
            format_raw_imags = ", ".join(f"{value:.2f}" for value in raw_imags)

            print(f"Step {n:03d} | RAW WARNING: {len(raw_imags)} imaginary freqs found: {format_raw_imags} cm-1")

        # Ideally a projected path should have ZERO imaginary frequencies left over
        proj_imags = [f for f in all_proj_freqs[n] if f < noise_floor]
        
        if len(proj_imags) > 0:
            issue_found = True
            # max_proj = min(proj_imags)
            format_proj_imags = ", ".join(f"{value:.2f}" for value in proj_imags)
            print(f"Step {n:03d} | PROJ WARNING: {len(proj_imags)} imaginary freqs remained after projection: {format_proj_imags} cm-1")

    if issue_found:
        print("-" * 60)
        print("DIAGNOSIS: Imaginary frequencies detected.")
        print("This usually indicates a mismatch between the geometric path tangent")
        print("and the electronic mass-weighted Hessian (e.g., a Cartesian NEB path).")
    else:
        print("All clear! No imaginary frequencies detected.")
    print("="*60 + "\n")








def diagnose_neb_path_health(coords,
                             gradients,
                             align=True, # [TODO] Remove after tests
                             trust_radius=0.2, 
                             kink_max_angle=0.99, 
                             tangent_max_shift=0.1,
                             plot_diag=False,
                             save_fig=''):
    """
    Diagnoses a Cartesian NEB path. 
    Automatically selects the best available coordinates and skips gradient checks if missing.
    """
    print("\n" + "="*60)
    print("CARTESIAN NEB PATH DIAGNOSTICS")
    print("="*60)

    # # get the coordinates (ir could be argued that diagnostics of raw coordinates is not useful, but...)
    # if len(self.proj_coordinates) > 0:
    #     coords = self.proj_coordinates
    #     print("Using projected coordinates for diagnostics.")
    # elif len(self.raw_coord) > 0:
    #     coords = self.raw_coord
    #     print("Using raw coordinates for diagnostics.")
    # else:
    #     print("ERROR: No coordinates found. Please load a path first.")
    #     return

    if len(coords) < 3:
        print("Path has fewer than 3 images. Diagnostics require a longer path.")
        return

    # This shouldn't be the place for alignment. It should be aligned before. 
    # This is just a server function, so diagnose whats given to you.
    # [TODO] Remove after tests
    # if align:
    #     print("Aligning path via Kabsch to remove global rotation/translation...")
    #     if len(self.masses) == 0 and len(self.atomic_numbers) > 0:
    #         self.masses = elem.get_masses(self.atomic_numbers)
    #         self.center_of_mass, self.rotational_matrices, self.transl_vectors, self.alignement_rmsd, self.aligned_coord = geom.align_coordinates(coords, self.masses)
    #     eval_coords = self.aligned_coord
    # else:
    eval_coords = coords

    # Metrics
    arc_lengths = _calc_arc_lengths(eval_coords)
    kinking = _calc_kinking_angles(eval_coords)
    tangent_shifts = _calc_tangent_consistency(eval_coords)

    # print report
    min_ds = np.min(arc_lengths)
    avg_ds = sum(arc_lengths) / len(arc_lengths)
    print(f"\nArc Lengths (Bohr):")
    print(f"Min arc length: {min_ds:.4f} | Avg arc length: {avg_ds:.4f} | max_move: {trust_radius:.4f}")
    if trust_radius > min_ds / 2.0:
            print(f"[WARNING] Trust radius (default max_move={trust_radius:.4f}) is LARGER \nthan the minimum arc length ({min_ds:.4f}).")

    min_kink = np.min(kinking)
    print(f"\nPath Kinking (cos \u03b8):") 
    print(f"  Worst kink: {min_kink:.3f} at image {np.argmin(kinking) + 1}")
    if min_kink < kink_max_angle:
        print("[WARNING] Kinking detected (cos \u03b8 < 0.98).")
        print("Tangent projection may fail to cleanly remove the reaction coordinate,")
        print("leaving residual imaginary frequencies in the projected Hessian.")

    max_tangent_shift = np.max(tangent_shifts)
    print(f"\nTangent Consistency:")
    print(f"  Max shift: {max_tangent_shift:.3f} between images {np.argmax(tangent_shifts)} and {np.argmax(tangent_shifts)+1}")
    if max_tangent_shift > tangent_max_shift:
        print("  -> [WARNING] Erratic tangent evolution detected. Projected frequencies may be unstable.")

    # Gradient check (needs the gradient of each image, so can only be done after engrad or freq calculation)
    if len(gradients) > 0:
        grad_overlaps = _calc_gradient_overlaps(gradients)
        max_overlap = np.max(grad_overlaps)
        print(f"\nGradient Overlap:")
        print(f"  Max overlap: {max_overlap:.4f} between images {np.argmax(grad_overlaps)} and {np.argmax(grad_overlaps)+1}")
        if max_overlap > 0.995:
            print("  -> [WARNING] Highly redundant images detected (overlap > 0.995). Path may be over-saturated.")
    else:
        print("\nGradient Overlap:")
        print("  -> [SKIPPED] No gradients loaded in the class yet.")
    
    print("="*60 + "\n")
    
    # Plotting if flagged
    if plot_diag:
        plotpath.plot_path_diagnostics(arc_lengths, kinking, tangent_shifts, 
                                    trust_radius=trust_radius, title_prefix="Cartesian NEB",
                                    save_fig=save_fig)








def diagnose_mwneb_path_health(coords,
                               gradients,
                               trust_radius=0.2, 
                                kink_max_angle=0.99, 
                                tangent_max_shift=0.1,
                                plot_diag=False,
                                save_fig=''):
    """
    Diagnoses a Mass-Weighted NEB path after optimization.
    Uses the internal mass-scaled coordinates and gradients.
    """
    print("\n" + "="*60)
    print("MASS-WEIGHTED NEB PATH DIAGNOSTICS")
    print("="*60)

    if len(coords) < 3:
        print("Mass-scaled coordinates missing or path too short. Run build_reaction_path() first.")
        return

    eval_coords = coords

    # Metrics
    arc_lengths = _calc_arc_lengths(eval_coords)
    kinking = _calc_kinking_angles(eval_coords)
    tangent_shifts = _calc_tangent_consistency(eval_coords)

    # pirnt report
    min_ds = np.min(arc_lengths)
    avg_ds = sum(arc_lengths) / len(arc_lengths)
    print(f"\nArc Lengths (Bohr):")
    print(f"Min arc length: {min_ds:.4f} | Avg arc length: {avg_ds:.4f} | max_move: {trust_radius:.4f}")
    if trust_radius > min_ds / 2.0:
            print(f"[WARNING] Trust radius (default max_move={trust_radius:.4f}) is LARGER \nthan the minimum arc length ({min_ds:.4f}).")

    min_kink = np.min(kinking)
    print(f"\nMW Path Kinking (cos \u03b8):")
    print(f"  Worst kink: {min_kink:.3f} at image {np.argmin(kinking) + 1}")
    if min_kink < kink_max_angle:
        print("[WARNING] Severe kinking detected in MW space. Tangent evaluation may be unstable.")

    max_tangent_shift = np.max(tangent_shifts)
    print(f"\nMW Tangent Consistency:")
    print(f"  Max shift: {max_tangent_shift:.3f} between images {np.argmax(tangent_shifts)} and {np.argmax(tangent_shifts)+1}")
    if max_tangent_shift > tangent_max_shift:
        print("[WARNING] Erratic tangent evolution detected in MW space.")

    # Gradient check (needs the gradient of each image, so can only be done after engrad or freq calculation)
    if len(gradients) > 0:
        grad_overlaps = _calc_gradient_overlaps(gradients)
        max_overlap = np.max(grad_overlaps)
        print(f"\nMW Gradient Overlap:")
        print(f"  Max overlap: {max_overlap:.4f} between images {np.argmax(grad_overlaps)} and {np.argmax(grad_overlaps)+1}")
        if max_overlap > 0.995:
            print("[WARNING] Redundant images detected in MW space.")
    else:
        print("\nMW Gradient Overlap:")
        print("[SKIPPED] No mass-weighted gradients loaded.")

    print("="*60 + "\n")
    
    # Plotting if flagged
    if plot_diag:
        plotpath.plot_path_diagnostics(arc_lengths, kinking, tangent_shifts, 
                                    trust_radius=trust_radius, title_prefix="Mass-Weighted NEB",
                                    save_fig=save_fig)

