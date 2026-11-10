import numpy as np
from tunkit.core import constants as const
from tunkit.solvers import tunnel_results
from tunkit.solvers import general
from scipy.optimize import curve_fit, root_scalar, brentq
from scipy import linalg 
import mpmath as mp
from scipy.integrate import trapezoid as trapz
from scipy.sparse import diags
from decimal import Decimal, getcontext
# Number of precision digits
getcontext().prec = 30
from dataclasses import replace








def _aps_sym_potential(y, h, b):
    # trigonometric function
    sin_y = np.sin(y)
    tan_y = np.tan(y) # or sin_y / np.cos(y)
    return h * (tan_y**2) + b * (sin_y**2)








def _aps_asym_potential(y, h, b, a):
    hterm = h * np.tan(y)**2
    # c = \sqrt(h) but could be changed in the future
    cterm = np.sqrt(h) * np.sin(y)/(np.cos(y)**2)
    bterm = b * np.sin(y)**2
    aterm = a * np.sin(y)
    return  hterm + cterm - bterm + aterm









def _unitless_var_calc(distances, energies, L, red_mass):
    """[x(array), energies(array), L, reduce_mass]
    Receives two arrays, the x distance array and the energies array,
    and returns them unitless, using the L parameter and the reduced
    mass mu."""
    # \epsilon = (8 * mu * L^2 * E) / (hbar^2 * pi^2)         
    # y = (pi * x) / (2 * L)

    energies_Dless = (8.0 * red_mass * L**2 * energies) / (const.hbar_au**2 * np.pi**2) #  * const.amu Sanitized to au!
    distances_Dless = (np.pi * distances) / (2.0 * L)
    return distances_Dless, energies_Dless








# TODO: decrease the number of parameters. They should come as much as possible from class variables
def calc_symmetric_energy_levels(path,
                                red_mass=1.0,
                                n_iter=100, 
                                m_step=10.0, 
                                n_levels=8,
                                prec_digits=60,
                                mi=1.0, 
                                hi=1.0, 
                                bi=-41.0, 
                                Li=0.8e-10,
                                matrix_size=1000,
                                raw_energies=False,
                                find_well=False,
                                use_mpmath=False,
                                kinetic_rate=False,
                                verbose=False,
                                verbose_L_fitting=False):

    # if kinetic_rate is not None:
    #     self.kinetic_rate = kinetic_rate

    red_mass = red_mass * const.amu_to_au


    react_path_m = path.reaction_path 

    if raw_energies:
        energies = path.energies
        print("Using nonprojected energies, mind you...")
    else:
        energies = path.proj_energies 
    
    # Zeroing in the barrier top
    energies = energies - np.max(energies)

    # Dynamically set initial L in mass-weighted Bohr if less then the buffer
    min_required_L = np.max(np.abs(react_path_m)) * 1.2 # 20% buffer. Should get h to around 10. Small h has some issues.
    if Li < min_required_L:
        if verbose:
            print(f"Warning: Initial L ({Li:.3e}) is smaller than path extent ({min_required_L:.3e}). Increasing L to minimum requirement.")
        Li = min_required_L

    m_vec, h_vec, b_vec, L_vec, res = [], [], [], [], []
    m, h, b, L = mi, hi, bi, Li

    # TODO: shouldn't use range(0, n_iter). The fitting should be done increasing m until the difference between
    # subsequent splittings is smaller than a threshold, or when it increases or changes too much.
    # Calculating the tunneling fitting for multiple m values
    old_split = 100
    spinner = ["|", "/", "-", "\\"]
    for i in range(0,n_iter):
        if not verbose_L_fitting and verbose:
            print("\rCalculating... " + spinner[i % 4], end="", flush=True)
        # old_m, old_h, old_b, old_L = m, h, b, L
        if find_well:
            p1 = np.array([react_path_m[0], energies[0]])
            hTarget = m**2 - 0.25
            h, b, L = _calculate_potential_parameters_stable(p1[0], p1[1], hTarget, red_mass)
        else:
            h, b, L = IRC_L_fit(react_path_m, energies, m, h, b, L, red_mass)

        raw_lambdas = solve_oblate_spectral(-b, 
                                            m, 
                                            matrix_size=matrix_size, 
                                            use_mpmath=use_mpmath, 
                                            n_levels=n_levels,
                                            prec_digits=prec_digits)



        epsilon_level = np.array(raw_lambdas) + 0.5 - (m**2)

        # convert factor (8*red_mass*amu*L**2/(hbar**2*np.pi**2)*h_plank)
        c_factor = (const.hbar_au**2*np.pi**2) / (8*red_mass*L**2)
        split = abs(epsilon_level[0] - epsilon_level[1]) * c_factor * const.hartree_to_cm
        diff_split = split - old_split
        
        if verbose_L_fitting:
            print("m = {:d}, h = {:.4e}, b = {:.3e}, L = {:.5e}, split(cm-1) = {:.5e}".format(int(m), 
                                                                                                    h, 
                                                                                                    b, 
                                                                                                    L, 
                                                                                                    float(split)))

        m_vec.append(m)
        h_vec.append(h)
        b_vec.append(b)
        L_vec.append(L)
        res.append(epsilon_level)
                
        m = m + m_step

        old_split = split

    if verbose:
        print('')
    m_vec = np.array(m_vec)
    h_vec = np.array(h_vec)
    b_vec = np.array(b_vec)
    L_vec = np.array(L_vec)
    aps_energy_levels = np.array(res)

    h_final = h_vec[-1]
    b_final = b_vec[-1]

    # Create a column vector of L^2 for broadcasting: shape (n_iter, 1)
    L_squared = L_vec[:, np.newaxis]**2
    
    scaling_factors = (const.hbar_au**2 * np.pi**2) / (8 * red_mass * L_squared)
    
    # Broadcast multiplication: (n_iter, n_levels) * (n_iter, 1)
    energy_levels = aps_energy_levels * scaling_factors

    freq_s = 0.0
    splitting = 0.0

    if len(energy_levels) > 0 and len(energy_levels[-1]) >= 2:

        splitting = energy_levels[-1][1] - energy_levels[-1][0]

        # Frequency in s^-1 = DeltaE / h_planck
        # DeltaE = hartree , conver to J then divide by h_planck
        freq_s = (splitting * const.hartree_to_J / const.h_plank) # Spectroscopic (full period)

        if kinetic_rate:
            tunneling_frequency = 2 * freq_s
        else:
            tunneling_frequency = freq_s

    # building the coherences to pass to the result class
    coherences = calculate_symmetric_level_coherences(h_vec[-1], b_vec[-1], n_levels=n_levels)

    leak = []
    theta = []
    for level in coherences:
        leak.append(level['leakage_prob'] * 100)
        theta.append(level['theta_deg'])

    # find the minimum of the dimensionless potential: h*tan^2(y) + b*sin^2(y)
    # scan a grid to find the minimum value
    # It's not as precise, but it's ok for only printing
    # The error is estimated to be ~0.01% with 2000 points
    y_scan = np.linspace(-np.pi/2 + 1e-3, np.pi/2 - 1e-3, 2000)
    u_scan = h_final * (np.tan(y_scan)**2) + b_final * (np.sin(y_scan)**2)
    u_min_dim = np.min(u_scan)# This will be negative (relative to barrier top)

    final_scaling_factor = scaling_factors[-1][0]
    well_depth_h = u_min_dim * final_scaling_factor

    results = tunnel_results.AnalyticSymResult(
            aps_energy_levels      = aps_energy_levels,
            energy_levels          = energy_levels,
            energy_levels_J        = energy_levels * const.hartree_to_J,
            energy_levels_cm       = energy_levels * const.hartree_to_cm,
            energy_levels_kcalmol  = energy_levels * const.hartree_to_kcalmol,
            leakage_percent        = leak,
            theta_deg              = theta,
            coherences             = coherences,
            m_vec                  = m_vec,
            h_vec                  = h_vec,
            b_vec                  = b_vec,
            L_vec                  = L_vec,
            splitting_hartree      = splitting,
            splitting_J            = splitting * const.hartree_to_J,
            splitting_cm           = splitting * const.hartree_to_cm,
            tunneling_frequency    = tunneling_frequency,
            kinetic_rate           = kinetic_rate,
            well_depth_hartree     = well_depth_h,
    )

    if verbose and len(energy_levels) > 0:
        
        print("-" * 90)
        print(f"{'Level':<5} | {'Energy (J)':<15} | {'Energy (cm^-1)':<15} | {'Leakage Prob (%)':<16} | {'Theta (deg)':<10}")
        print("-" * 90)
        for i, (eh, ec) in enumerate(zip(energy_levels[-1], energy_levels[-1]*const.hartree_to_cm)):
            if i >= n_levels: # print only the levels the user want
                break

            # This sets the bottom of the well to 0.
            e_shifted_h = eh - well_depth_h
            e_shifted_cm = e_shifted_h * const.hartree_to_cm
            e_shifted_J = e_shifted_h * const.hartree_to_J

            print(f"{i:<5} | {float(e_shifted_J):>15.4e} | {float(e_shifted_cm):>15.2f} | {leak[i]:>16.4f} | {theta[i]:>10.4f}")
        print("-" * 90)
        if verbose_L_fitting:
            print(f"Parameters final values:")
            print(f"m = {int(m_vec[-1]):d}, h = {h_vec[-1]:.4e}, b = {b_vec[-1]:.3e}, L = {L_vec[-1]:.5e}")
        print(f"Ground state splitting (J): {float(splitting*const.hartree_to_J):.4e} J")
        print(f"Ground state splitting (cm-1): {float(splitting*const.hartree_to_cm):.4e} cm-1")
        # print(f"Ground state splitting: {float(freq_s/const.speed_light):.4e} cm-1")
        print(f"Associated Frequency:   {float(freq_s):.4e} s^-1")

    return results








def calculate_symmetric_level_coherences(h, b, n_levels):
    """
    Calculates the coherence (mixing angle and leakage probability) 
    for the symmetric energy levels using a finite-difference grid.
    """
    # Commented after refactoring. Remove after testing.
    # if not hasattr(self, 'h_vec') or len(self.h_vec) == 0:
    #     return []

    # h = self.h_vec[-1]
    # b = self.b_vec[-1]
    
    # if n_levels is None:
    #     raise ValueError ('n_levels has to be ')
    #     n_levels = len(self.energy_levels[-1])

    # grid
    N_grid = 2000
    epsilon = 1e-3
    y_grid = np.linspace(-np.pi/2 + epsilon, np.pi/2 - epsilon, N_grid)
    dy = y_grid[1] - y_grid[0]
    
    # [FIXED] reuse method symmetric potential. Remove after testing
    # U_dim = h * (np.tan(y_grid)**2) + b * (np.sin(y_grid)**2)
    U_dim = _aps_sym_potential(y_grid,h, b)
    U_dim = np.clip(U_dim, -1e9, 1e9)
    
    kin_coeff = 1.0 / (dy**2)
    diag = 2 * kin_coeff + U_dim
    off_diag = -kin_coeff * np.ones(N_grid - 1)
    
    evals_dim, evecs = linalg.eigh_tridiagonal(
        diag, off_diag, select='i', select_range=(0, n_levels-1)
    )
    
    # symmetric DWP barrier is in 0.0 by definition
    barrier_idx = np.argmin(np.abs(y_grid - 0.0))

    coherences = []
    for i in range(len(evals_dim)):
        # normalize wasvefunction
        psi = evecs[:, i]
        norm = np.sqrt(trapz(y=psi**2, x=y_grid))
        psi = psi / norm
        
        prob_density = psi**2
        
        prob_left = trapz(y=prob_density[:barrier_idx], x=y_grid[:barrier_idx])
        prob_right = trapz(y=prob_density[barrier_idx:], x=y_grid[barrier_idx:])
        
        leakage_prob = min(prob_left, prob_right)
        theta_rad = np.arcsin(np.sqrt(max(0, leakage_prob)))
        
        coherences.append({
            'level': i,
            'leakage_prob': leakage_prob,
            'theta_deg': np.degrees(theta_rad)
        })
        
    return coherences








def solve_oblate_spectral(b, m_val, n_levels=2, matrix_size=500, use_mpmath=True, prec_digits=60):
    """
    Solves the Oblate Spheroidal eigenvalues.
    
    Args:
        n_levels (int): The number of energy levels to return (e.g., 2, 4, 10).
        use_mpmath (bool): If True, uses arbitrary precision arithmetic to resolve tiny splittings.
        prec_digits (int): Number of decimal digits of precision (try 60 or 100).
    """
    if not use_mpmath:
        # numpy implementation, it's fast but can't get very small splittings (looking at you trimethylamine)
        c_sq = b
        m_val = float(m_val)
        
        def get_vals_numpy(start_index):
            N = matrix_size
            js = np.arange(N)
            n_degrees = m_val + start_index + 2 * js
            
            term_c = (2*n_degrees*(n_degrees+1) - 2*m_val**2 - 1) / ((2*n_degrees - 1)*(2*n_degrees + 3))
            diag = n_degrees * (n_degrees + 1) - c_sq * term_c 
            
            n_sub = n_degrees[:-1]
            num = (n_sub + 2 + m_val) * (n_sub + 2 - m_val) * (n_sub + 1 + m_val) * (n_sub + 1 - m_val)
            den = (2*n_sub + 1) * (2*n_sub + 5) * (2*n_sub + 3)**2
            
            safe_array = np.maximum(0, num / den) # [FIXED] Clamp small negatives to 0, then take sqrt
            off_diag = -c_sq * np.sqrt(safe_array)
            
            vals, _ = linalg.eigh_tridiagonal(diag, off_diag)
            return vals

        # get all eigenvalues for even and odd blocks
        vals_even = get_vals_numpy(0)
        vals_odd = get_vals_numpy(1)
        
        # interleave them: Even0, Odd0, Even1, Odd1...
        combined = []
        for k in range(n_levels):
            if k % 2 == 0:
                combined.append(vals_even[k // 2])
            else:
                combined.append(vals_odd[k // 2])
        return combined

    else:
        # mpmath implementation, huge number of precision digits
        mp.mp.dps = prec_digits
        
        m_mp = mp.mpf(m_val)
        c_sq = mp.mpf(b)
        
        # helper to generate matrix elements for a specific block (Even=0 / Odd=1)
        def get_matrix_elements(start_index):
            diag = []
            off_diag_sq = [] # Store squared off-diagonal for Sturm calculation
            
            for j in range(matrix_size):
                n = m_mp + start_index + 2*j
                
                # diagonal
                term_c = (2*n*(n+1) - 2*m_mp**2 - 1) / ((2*n - 1)*(2*n + 3))
                d_val = n*(n+1) - c_sq * term_c
                diag.append(d_val)
                
                # off-diagonal (squared)
                if j < matrix_size - 1:
                    n_sub = n
                    num = (n_sub + 2 + m_mp) * (n_sub + 2 - m_mp) * (n_sub + 1 + m_mp) * (n_sub + 1 - m_mp)
                    den = (2*n_sub + 1) * (2*n_sub + 5) * (2*n_sub + 3)**2
                    # We only need the square for Sturm sequence, which avoids one sqrt()
                    od_sq = (c_sq**2) * (num / den) 
                    off_diag_sq.append(od_sq)
            return diag, off_diag_sq

        # Sturm Sequence: Count eigenvalues smaller than sigma
        def count_below(sigma, diag, off_diag_sq):
            count = 0
            q = diag[0] - sigma
            if q < 0: count += 1
            
            for i in range(1, len(diag)):
                if q == 0: q = mp.mpf('1e-50')
                # q_i = (d_i - sigma) - e_{i-1}^2 / q_{i-1}
                q = (diag[i] - sigma) - off_diag_sq[i-1]/q
                if q < 0: count += 1
            return count

        # find the k-th eigenvalue using Bisection
        def find_eigenvalue_k(k, diag, off_diag_sq):
            # Gerschgorin bounds estimate
            # max off-diagonal estimate (approx sqrt of max sq)
            max_off = mp.sqrt(max(off_diag_sq)) if off_diag_sq else 0
            low = min(diag) - max_off - 1
            high = max(diag) + max_off + 1
            
            # Bisection
            # we stop when the interval size is smaller than 10^-dps
            # or a fixed number of iterations
            for _ in range(250):
                mid = (low + high) / 2
                if count_below(mid, diag, off_diag_sq) <= k:
                    low = mid
                else:
                    high = mid
            return (low + high) / 2

        # Pre-calculate matrix elements for both blocks
        diag_even, off_sq_even = get_matrix_elements(0)
        diag_odd, off_sq_odd = get_matrix_elements(1)
        
        combined_results = []
        
        # request specific levels
        # physical level 0 -> Even block, index 0
        # physical level 1 -> Odd block, index 0
        # physical level 2 -> Even block, index 1
        # ...
        
        for i in range(n_levels):
            target_index = i // 2
            
            if i % 2 == 0:
                # even Parity
                val = find_eigenvalue_k(target_index, diag_even, off_sq_even)
            else:
                # odd Parity
                val = find_eigenvalue_k(target_index, diag_odd, off_sq_odd)
            
            combined_results.append(val)
            
        return combined_results








def IRC_L_fit(X, energies, m_target, hi, bi, Li, red_mass, Lfit_N_iter=1000, Lfit_tol=1e-4, verbose=False):
    # the value for h we need to reach
    h_target = m_target**2 - 1/4
    
    udist, uenerg = _unitless_var_calc(X, energies, Li, red_mass) # transforming to dimentionless units
    param, _ = curve_fit(_aps_sym_potential, udist, uenerg, [hi, bi], ftol=1e-15, xtol=1e-15) #extracting h b
    h, b = param # setting inital parameters h and b
    
    dh = float(h_target - h) # 0.75 - 0.54 = 0.18
    if abs(dh) < Lfit_tol: # First test, if the initial guess is already good, we return it
        return h, b, Li 
    
    step = ( Li / 10 ) * np.sign(dh) # if the test fails, we set a step for L
    L = Li + step # and modify L with the step

    for _ in range(0,Lfit_N_iter): # Try Lfit_N_iter times
        old_step = np.sign(step) 
        
        udist, uenerg = _unitless_var_calc(X, energies, L, red_mass) # Transform to dimensionless units with the new L
        param, _ = curve_fit(_aps_sym_potential, udist, uenerg, [h, b], ftol=1e-15, xtol=1e-15)
        h, b = param # New h and b parameters
        dh = float(h_target - h)
        step = abs(step) * np.sign(dh)

        if abs(dh) < Lfit_tol: # test if the threshold was reached
            return h, b, L #, result

        if old_step != np.sign(step): # Check if we went past the h_target
            step = step / 3

        L = L + step # update L
    if verbose:                 # Final check for physical validity
        if h < 0.73: 
            print(f"The fitting did not converge properly, try another combination of initial values for h, b and L. h = {h}")
            raise ValueError('The fitting did not converge properly.')
        print(f"Tolerance was not reached with {Lfit_N_iter} steps. Final result: dh = {dh:.5f} and h = {h} h_target = {h_target}")
    return h, b, L








# It is faster, but not as reliable. My og function can find L from almost any value. This one has to be within 40% of
# the correct value. Leaving it here for perhaps future reference.
def IRC_L_fit_alt(X, energies, m_target, hi, bi, Li, red_mass, Lfit_tol=1e-4, max_iter=50):
    h_target = m_target**2 - 0.25
    
    L = Li
    h, b = hi, bi
    
    # Define a limit for how much L can change in one step (to keep curve_fit stable)
    # 5% change maximum per iteration should prevent bad fits
    max_step_pct = 0.05 

    for i in range(max_iter):

        udist, uenerg = _unitless_var_calc(X, energies, L, red_mass)
        try:

            popt, _ = curve_fit(_aps_sym_potential, udist, uenerg, p0=[h, b], ftol=1e-15, xtol=1e-15)
            h, b = popt
        except RuntimeError:
            print(f"Fit failed at L={L}")
            break 

        diff = h - h_target
        
        # Check convergence
        if abs(diff) < Lfit_tol:
            # check if convergence is going anywhere. WHen there is a problem h tends to 0
            if h < 0.73:
                print(f"Warning: Converged to h={h:.4f} < 0.73. Result is probably unphysical.")
            return h, b, L

        # Calculate Gradient (Numerical Derivative) to find smart direction
        # We perturb L slightly (0.1%) to see how h changes
        dL_probe = L * 0.001 
        L_probe = L + dL_probe
        
        udist_p, uenerg_p = _unitless_var_calc(X, energies, L_probe, red_mass)
        try:
            popt_p, _ = curve_fit(_aps_sym_potential, udist_p, uenerg_p, p0=[h, b], ftol=1e-15, xtol=1e-15)
            h_probe, _ = popt_p
            
            # Slope dh/dL
            slope = (h_probe - h) / dL_probe
        except:
            slope = 0 # Fallback to avoid crash

        # Newton Step: delta_L = - error / slope
        if abs(slope) > 1e-12:
            step = - diff / slope
        else:
            # If slope is flat, fallback to simple small step 
            step = L * 0.01 * np.sign(-diff)

        # DAMPING / CLAMPING 
        # Do not allow L to change by more than max_step_pct
        # This keeps L inside the "basin of attraction" for curve_fit
        max_step = L * max_step_pct
        
        if abs(step) > max_step:
            step = max_step * np.sign(step)

        # Safety check for minimum L (cannot be smaller than max(X))
        min_L = np.max(np.abs(X)) * 1.01
        if (L + step) < min_L:
            step = (min_L - L) / 2 # move halfway to the limit

        L = L + step
        
    print(f"Tolerance not reached in {max_iter} steps. Final diff: {diff}")
    return h, b, L







def _calculate_potential_parameters_stable(x_phys, E_phys, h_target, mu):
    """Calculates L and the dimensionless potential parameters h and b.

    This function acts as a wrapper that uses a numerically stable method
    to find the length scale L that maps a given physical minimum (x_phys, E_phys)
    to a dimensionless double-well potential characterized by a target h-parameter.
    It returns L along with the self-consistent h and b parameters.

    Args:
        x_phys (float): The physical x-coordinate of the minimum (in bohrs).
        E_phys (float): The physical energy of the minimum (in hartree).
        h_target (float): The target h-parameter for the dimensionless potential.
        mu (float): The effective mass of the particle (in atomic mass units).

    Returns:
        list(float) | None: A list containing the calculated 'L', 'h',
                                and 'b' parameters. The returned 'h' is recalculated
                                from the found L for consistency, and it should match
                                the h_target to a high precision. Returns None if
                                the solver fails.
    """
    tau_found, L_found = _find_tau_and_L_stable(x_phys, E_phys, h_target, mu)

    if L_found is None:
        print("Could not determine a valid solution. Aborting.")
        return None
    
    # Recalculate epsilon using the found L to check consistency
    epsilon = (8 * mu * L_found**2 * E_phys) / (const.hbar_au**2 * np.pi**2)

    # Recalculate h from the results
    h_recalculated = -epsilon / (np.tan(tau_found)**4)
    b_calculated = -h_recalculated / (np.cos(tau_found)**4)
    
    return h_recalculated, b_calculated, L_found








def _find_tau_and_L_stable(x_phys, E_phys, h_target, mu):
    """Solves for the dimensionless coordinate tau and the length scale L.

    This function sets up a numerically stable objective function in terms of tau (τ)
    and solves for its root in the physical domain (0, pi/2). It then uses the
    found tau to calculate the corresponding length scale L.

    Args:
        x_phys (float): The physical x-coordinate of the minimum (in meters).
        E_phys (float): The physical energy of the minimum (in Joules).
        h_target (float): The target h-parameter for the dimensionless potential,
                        used to set up the equation.
        mu (float): The effective mass of the particle (in kg).

    Returns:
        tuple[float | None, float | None]: A tuple containing the found tau and L.
                                        Returns (None, None) if the solver fails.
    """

    # define the stable objective function in terms of tau
    def objective_function_tau(tau):
        # physical domain for tau is (0, pi/2)
        if tau <= 0 or tau >= np.pi / 2:
            return np.inf
        term1 = (2 * mu * x_phys**2 * E_phys) / (const.hbar_au**2 * tau**2)
        term2 = h_target * np.tan(tau)**4
        return term1 + term2

    # bracket for tau is always (0, pi/2)
    # use a small epsilon to avoid division by zero at tau=0
    lower_bound_tau = 1e-9
    upper_bound_tau = np.pi/2 - 1e-9

    try:
        sol = root_scalar(
            objective_function_tau,
            method='brentq',
            bracket=[lower_bound_tau, upper_bound_tau]
        )
        if sol.converged:
            tau_found = sol.root
            # calculate the corresponding L
            L_found = (np.pi * x_phys) / (2 * tau_found)
            return tau_found, L_found
        else:
            return None, None
    except ValueError:        # This can happen if the function doesn't cross zero in the bracket
        return None, None








def calc_asymmetric_energy_levels(path, 
                                    hi=50.0, 
                                    bi=150.0, 
                                    ai=0.0, 
                                    Li=1, #0.9e-10, 
                                    n_iter=100, 
                                    L_step=0.2,#0.02e-10, 
                                    red_mass=1.0,
                                    raw_energies=False,
                                    find_well=False,
                                    n_levels = 6,
                                    kinetic_rate=None,
                                    verbose=True,
                                    verbose_L_fitting=False):
    """
    Wrapper to solve the asymmetric potential and convert results to physical units.
    """

    # if kinetic_rate is not None:
    #     self.kinetic_rate = kinetic_rate

    # kinetic_rate = path.kinetic_rate

    # Used to convert the displacements from bohrs to meters
    # Since the view won't change anything, maybe use the class variable?
    react_path_m = path.reaction_path 

    red_mass = red_mass * const.amu_to_au

    if raw_energies:
        energies = path.energies 
        print("Using nonprojected energies...")
    else:
        energies = path.proj_energies 
    
    # zeroing
    energies = energies - np.max(energies)

    h_vec = []
    b_vec = []
    a_vec = []
    L_vec = []
    res = []
    vectors = []
    
    h, b, a, L = hi, bi, ai, Li        


    min_required_L = np.max(np.abs(react_path_m)) * 1.2 # 20% buffer. Should get h to around 10. Small h has some issues.
    if L < min_required_L:
        if verbose:
            print(f"Warning: Initial L ({L:.3e}) is smaller than path extent ({min_required_L:.3e}). Increasing L to minimum requirement.")
        L = min_required_L

    for i in range(n_iter):

        udist, uenergies = _unitless_var_calc(react_path_m, energies, L, red_mass)
        c0 = [h, b, a]
        try:
            param, cov = curve_fit(_aps_asym_potential, udist, uenergies, c0, ftol=1e-15, xtol=1e-15)
        except RuntimeError:
            print(f"Fit failed at iteration {i} with L={L}")
            break

        h_fit, b_fit, a_fit = param

        if h_fit < 0.73:
            if verbose:
                print(f"Warning: Fitted h={h_fit:.4f} < 0.73 at iter {i}. Check initial values.")
                raise ValueError('Fitted h is too small!')

        eig_vals, eig_vecs, grid = solve_asymmetric_finite_difference(h_fit, b_fit, a_fit, n_levels=n_levels)

        splitting_mp = abs(eig_vals[0] - eig_vals[1])

        split = splitting_mp / (8*red_mass*L**2/(const.hbar_au**2*np.pi**2)) * const.hartree_to_J / const.h_plank
        
        if verbose_L_fitting:
            print("h = {:.4e}, b = {:.4e}, a = {:.4e}, L = {:.5e}, tunneling frequency = {:.5e}".format(h_fit, 
                                                                                                    b_fit, 
                                                                                                    a_fit, 
                                                                                                    L, 
                                                                                                    float(split)))

        h_vec.append(h_fit)
        b_vec.append(b_fit)
        a_vec.append(a_fit)
        L_vec.append(L)
        res.append(eig_vals)
        vectors.append(eig_vecs)

        h, b, a = h_fit, b_fit, a_fit
        L += L_step

    h_vec = np.array(h_vec)
    b_vec = np.array(b_vec)
    a_vec = np.array(a_vec) 
    L_vec = np.array(L_vec)

    # energy levels in dimensionless units
    # shape: (n_iter, n_levels)
    che_energy_levels = np.array(res)

    # Create a column vector of L^2 for broadcasting: shape (n_iter, 1)
    L_squared = L_vec[:, np.newaxis]**2
    
    scaling_factors = (const.hbar_au**2 * np.pi**2) / (8 * red_mass * L_squared)
    
    energies_hartree = che_energy_levels * scaling_factors
    energies_joules = energies_hartree * const.hartree_to_J
    
    # hartree
    energy_levels = energies_hartree

    freq_s = 0.0
    splitting = 0.0
    
    if len(energies_joules) > 0 and len(energies_joules[-1]) >= 2:
        e0 = energies_joules[-1][0]
        e1 = energies_joules[-1][1]
        splitting = e1 - e0
        
        # Frequency in s^-1 = DeltaE / h_planck
        freq_s = (splitting / const.h_plank) # Spectroscopic (full period)

        if kinetic_rate:
            tunneling_frequency = 2 * freq_s
        else:
            tunneling_frequency = freq_s


    # Only calculate the coherences if they are asked by the user
    coherences = calculate_level_coherences(h_vec[-1], # final point h, b and a
                                            b_vec[-1],
                                            a_vec[-1],
                                            n_levels=len(energies_joules[-1])) 

    leak = []
    theta = []
    for level in coherences:
        leak.append(level['leakage_prob'] * 100)
        theta.append(level['theta_deg'])


    results = tunnel_results.AnalyticAsymResult(
            che_energy_levels      = che_energy_levels,
            energy_levels          = energy_levels,
            energy_levels_J        = energy_levels * const.hartree_to_J,
            energy_levels_cm       = energy_levels * const.hartree_to_cm,
            energy_levels_kcalmol  = energy_levels * const.hartree_to_kcalmol,
            leakage_percent        = leak,
            theta_deg              = theta,
            coherences             = coherences,
            a_vec                  = a_vec,
            h_vec                  = h_vec,
            b_vec                  = b_vec,
            L_vec                  = L_vec,
            splitting_hartree      = splitting,
            splitting_J            = splitting * const.hartree_to_J,
            splitting_cm           = splitting * const.hartree_to_cm,
            tunneling_frequency    = tunneling_frequency,
            kinetic_rate           = kinetic_rate,
    )

    if verbose and len(energies_joules) > 0:
        
        print("-" * 75)
        print(f"{'Level':<5} | {'Energy (J)':<15} | {'Energy (cm^-1)':<15} | {'Leakage Prob (%)':<16} | {'Theta (deg)':<10}")
        print("-" * 75)
        for i, (ej, ec) in enumerate(zip(energies_joules[-1], energies_hartree[-1]*const.hartree_to_cm)):
            
            leak = coherences[i]['leakage_prob']
            theta = coherences[i]['theta_deg']
            
            print(f"{i:<5} | {ej:<15.4e} | {ec:<15.2f} | {leak*100:<16.2f} | {theta:<10.4f}")

        print("-" * 75)
        
        print(f"Ground state splitting: {splitting:.4e} J")
        print(f"Ground state splitting: {freq_s/const.speed_light:.4e} cm-1")
        print(f"Associated Frequency:   {freq_s:.4e} s^-1")
        
        if a_vec[-1] != 0:
            print("Note: In asymmetric wells, splitting includes the asymmetry offset (detuning).")

    return results








# [TODO] change the name to add asymmetric to it...
def calculate_level_coherences(h, b, a, n_levels=None):
    """
    Calculates the coherence (mixing angle and leakage probability) 
    for the calculated asymmetric energy levels.
    """
    # if not hasattr(self, 'h_vec') or len(self.h_vec) == 0:
    #     print("No calculation results found. Run calc_asymmetric_energy_levels first.")
    #     return []

    # h = self.h_vec[-1]
    # b = self.b_vec[-1]
    # a = self.a_vec[-1]
    
    # if n_levels is None:
    #     n_levels = len(self.energy_levels[-1])

    # Recalculate grid and wave functions for the final parameters
    eig_vals, eig_vecs, y_grid = solve_asymmetric_finite_difference(h, b, a, n_levels=n_levels)

    # potential (careful on the sign)
    # [FIXED] reuse method symmetric potential. Remove after testing
    # tan_y = np.tan(y_grid)
    # sin_y = np.sin(y_grid)
    # cos_sq = np.cos(y_grid)**2
    # U_dim = (h * tan_y**2) + (c * sin_y / cos_sq) - (b * sin_y**2) + (a * sin_y)
    # c = np.sqrt(h)
    U_dim = _aps_asym_potential(y_grid, h, b, a)
    
    edge_buffer = len(U_dim) // 10  # exclude diverging edges

    # Find both wells (minima) in left and right halves of the interior
    interior_start = edge_buffer
    interior_end = len(U_dim) - edge_buffer
    midpoint = len(U_dim) // 2

    left_min_idx = interior_start + np.argmin(U_dim[interior_start:midpoint])
    right_min_idx = midpoint + np.argmin(U_dim[midpoint:interior_end])

    # Barrier is the maximum between the two wells
    barrier_idx = left_min_idx + np.argmax(U_dim[left_min_idx:right_min_idx])

    coherences = []
    for i in range(n_levels):
        psi = eig_vecs[:, i]    
        # Normalize on the actual grid (continuous normalization)
        norm = np.sqrt(trapz(y=psi**2, x=y_grid))
        psi = psi / norm
        
        prob_density = psi**2
        
        prob_left = trapz(y=prob_density[:barrier_idx], x=y_grid[:barrier_idx])
        prob_right = trapz(y=prob_density[barrier_idx:], x=y_grid[barrier_idx:])
        
        # leakage is the smaller one. Could check a better way of defining these.
        leakage_prob = min(prob_left, prob_right)
        
        # mixing angle theta = arcsin(sqrt(leakage_prob))
        # max(0, val) avoid negative float point error
        theta_rad = np.arcsin(np.sqrt(max(0, leakage_prob)))
        theta_deg = np.degrees(theta_rad)
        
        coherences.append({
            'level': i,
            'prob_left': prob_left,
            'prob_right': prob_right,
            'leakage_prob': leakage_prob,
            'theta_deg': theta_deg
        })
        
    # self.coherences = coherences
    return coherences








def solve_asymmetric_finite_difference(h, b, a, N=5000, n_levels=6, epsilon=1e-3):
    """
    Solves the Asymmetric Double Well using a Direct Tridiagonal Solver.
    Includes numerical stabilization for singular boundaries.
    """
    # grid, domain is (-pi/2, pi/2). 
    L_bound = np.pi / 2.0
    
    # [FIX] Increased epsilon from 1e-6 to 1e-3. avoid large values.
    # Changed to a paramneter to allow the user to change it.
    # epsilon = 1e-3 
    
    y = np.linspace(-L_bound + epsilon, L_bound - epsilon, N)
    dy = y[1] - y[0]
    
    
    # U = h tan^2 + sqrt(h) sin/cos^2 - b sin^2 + a sin
    c = np.sqrt(h) 
    
    # [TODO] Changed to method. Check results and remove it later.
    # tan_y = np.tan(y)
    # sin_y = np.sin(y)
    # cos_sq = np.cos(y)**2
    # U = (h * tan_y**2) + (c * sin_y / cos_sq) - (b * sin_y**2) + (a * sin_y)

    U = _aps_asym_potential(y, h, b, a)
    
    # [FIX] Clamp the potential.
    # This prevents floating point issues if the grid accidentally hits a singularity.
    # 1e8 should high enough to be an "infinite wall" for the physics, but low enough for math.
    U = np.clip(U, -1e8, 1e8)

    # construct tridiagonal matrix elements
    # the Hamiltonian is H = -d^2/dy^2 + U
    kin_coeff = 1.0 / (dy**2)
    
    d = 2 * kin_coeff + U      # main diagonal elements
    e = -kin_coeff * np.ones(N-1) # off-diagonal elements
    
    # diagonilize and get n_levels
    eigenvalues, eigenvectors = linalg.eigh_tridiagonal(
        d, e, select='i', select_range=(0, n_levels-1)
    )
    
    # notmalize
    for i in range(eigenvalues.shape[0]):
        norm = np.sqrt(trapz(eigenvectors[:, i]**2, x=y))
        eigenvectors[:, i] /= norm
        
    return eigenvalues, eigenvectors, y








def _generate_isolated_potentials(U, y):
    """
    Takes the full double well potential and splits it into 
    isolated left and right wells by replacing the opposite side 
    with a large numerical wall.
    """
    N = len(U)
    edge_buffer = N // 10
    midpoint = N // 2

    # getting minima indexes instead of the barrier
    left_min_idx = edge_buffer + np.argmin(U[edge_buffer:midpoint])
    right_min_idx = midpoint + np.argmin(U[midpoint:-edge_buffer])
    barrier_idx = left_min_idx + np.argmax(U[left_min_idx:right_min_idx])
    
    # making Left well
    U_left = np.copy(U)
    U_left[barrier_idx:] = 1e8  # Wipe out the right well
    
    # making right well
    U_right = np.copy(U)
    U_right[:barrier_idx] = 1e8 # Wipe out the left well
    
    return U_left, U_right, left_min_idx, right_min_idx, barrier_idx








def _solve_custom_potential(U, dy, n_levels):
    """
    A modular version of the finite difference solver that accepts a custom U.
    """
    N = len(U)
    kin_coeff = 1.0 / (dy**2)
    
    d = 2 * kin_coeff + U      # main diagonal elements
    e = -kin_coeff * np.ones(N-1) # off-diagonal elements
    
    # Diagonalizing
    eigenvalues, eigenvectors = linalg.eigh_tridiagonal(
        d, e, select='i', select_range=(0, n_levels-1)
    )
    
    # Normalizing
    for i in range(eigenvalues.shape[0]):
        norm = np.sqrt(trapz(eigenvectors[:, i]**2, dx=dy))
        eigenvectors[:, i] /= norm
        
    return eigenvalues, eigenvectors








def _calculate_tunneling_matrix(U_full, vecs_L, vecs_R, dy, n_levels):
    """
    Calculates the off-diagonal tunneling amplitudes t_{ij} between 
    localized left states and right states.
    """
    N = len(U_full)
    kin_coeff = 1.0 / (dy**2)
    
    # Construct the full Hamiltonian as a sparse matrix
    main_diag = 2 * kin_coeff + U_full
    off_diag = -kin_coeff * np.ones(N-1)
    H_full = diags([off_diag, main_diag, off_diag], [-1, 0, 1])
    
    # Initialize the amplitude matrix
    t_matrix = np.zeros((n_levels, n_levels))
    
    for i in range(n_levels):
        for j in range(n_levels):
            phi_L_i = vecs_L[:, i]
            phi_R_j = vecs_R[:, j]
            
            # Apply the full Hamiltonian to the right state: H |phi_R>
            H_phi_R = H_full.dot(phi_R_j)
            
            # Overlap with the left state: <phi_L | H | phi_R>
            t_ij = trapz(phi_L_i * H_phi_R, dx=dy)
            
            # Take the absolute value (the phase should be usually arbitrary here)
            t_matrix[i, j] = np.abs(t_ij)
            
    return t_matrix








def calculate_matrix_tunneling_rate(results, 
                                    starting_well='higher', # is there an use case for the opposite?
                                    theta_threshold=5.0,
                                    red_mass=1.0,
                                    temperature=3.0,
                                    harm_attempt_freq=False,
                                    verbose=True):
    """
    Calculates the irreversible tunneling rate from the higher well ground state 
    to all localized states in the global minimum well, assuming a T=temperature matrix bath.
    """

    # Converting the mass to au
    red_mass_au = red_mass * const.amu_to_au

    # Get the final parameters 
    h, b, a, L = results.h_vec[-1], results.b_vec[-1], results.a_vec[-1], results.L_vec[-1]

    # Run diagnostics to get coherence data and grid
    coherences = calculate_level_coherences(h, b, a, n_levels = len(results.energy_levels[-1]))
    
    # Re-solve to get the exact grid and dy
    _, _, y_grid = solve_asymmetric_finite_difference(h, b, a)
    dy = y_grid[1] - y_grid[0]
    
    # Build potential and clip to avoid numerical issues
    U_full = _aps_asym_potential(y_grid, h, b, a)
    U_full = np.clip(U_full, -1e8, 1e8)
    
    # generating the isolated potentials
    ( U_L, U_R, left_min_idx, 
        right_min_idx, barrier_idx ) = _generate_isolated_potentials(U_full, y_grid)
    
    # Determine which well is the higher one
    if U_full[left_min_idx] > U_full[right_min_idx]:
        reactant_well = 'left'
        reactant_min_idx = left_min_idx
    else:
        reactant_well = 'right'
        reactant_min_idx = right_min_idx
        
    # Filter states based on the mixing angle threshold
    reactant_states = []
    product_states = []
    V_min_reactant = U_full[left_min_idx] if reactant_well == 'left' else U_full[right_min_idx]

    for coh in coherences:
        if coh['theta_deg'] > theta_threshold:
            continue # [IMPORTANT] skipping delocalized states 
            
        # Classify based on where the probability majority lies
        if coh['prob_left'] > coh['prob_right']:
            if reactant_well == 'left':
                reactant_states.append(coh)
            else:
                product_states.append(coh)
        else:
            if reactant_well == 'right':
                reactant_states.append(coh)
            else:
                product_states.append(coh)
                
    if not reactant_states:
        raise ValueError("No localized ground state found in the reactant well!")
        

    # Compute the effective attempt frequency (omega_eff) from ZPE
    # E_L - V_min = 0.5 * hbar * omega_eff
    E_reactant_0 = results.energy_levels[-1][reactant_states[0]['level']] # in Hartree
    
    # Convert V_min from dimensionless to Hartree 
    scaling_factor = (const.hbar_au**2 * np.pi**2) / (8 * red_mass_au * L**2)

    # Map dimensionless variables to physical units
    x_phys = (2.0 * L / np.pi) * y_grid
    V_phys = U_full * scaling_factor
    
    if harm_attempt_freq:
        # Harmonic approximation for omega
        V_min_hartree = V_min_reactant * scaling_factor
        omega_eff_au = 2.0 * (E_reactant_0 - V_min_hartree) / const.hbar_au
    else:
        # Attempt frequency classically integrated (check)
        omega_eff_au = general.calculate_classical_frequency(
            x_array=x_phys, 
            V_array=V_phys, 
            E_target=E_reactant_0, 
            mass=red_mass_au, 
            min_idx=reactant_min_idx, 
            barrier_idx=barrier_idx, 
            reactant_well=reactant_well
        )
    
    # get only the relevant levels to avoid numerical issues
    max_level_needed = max([s['level'] for s in (reactant_states + product_states)]) + 1
    
    _, vecs_L = _solve_custom_potential(U_L, dy, n_levels=max_level_needed)
    _, vecs_R = _solve_custom_potential(U_R, dy, n_levels=max_level_needed)
    

    # Calculate amplitudes and total rate
    t_matrix = _calculate_tunneling_matrix(U_full, vecs_L, vecs_R, dy, n_levels=max_level_needed)
    
    total_rate_au = 0.0
    r_idx = reactant_states[0]['level'] # Starting from the reactant ground state
    
    au_to_inv_s = const.hartree_to_J / const.hbar

    if verbose:
        print(f"\n--- Matrix Tunneling Analysis (T={temperature}K) ---")
        print(f"Reactant well: {reactant_well.capitalize()}")
        print(f"Effective Attempt Frequency: {omega_eff_au * au_to_inv_s:.4e} s^-1")
        print("-" * 50)

    channel_index = []
    t_ij_array = []
    rate_ij_array = []
    delta_E_array = []
    thermal_penalties = []
    
    for p_state in product_states:
        p_idx = p_state['level']
        channel_index.append(p_idx)

        # product height in relation to the reactant, to check  viability
        E_product = results.energy_levels[-1][p_idx]
        delta_E_au = E_product - E_reactant_0
        delta_E_array.append(delta_E_au)
        
        # Extract t depending on direction
        if reactant_well == 'left':
            t_ij_au = t_matrix[r_idx, p_idx] * scaling_factor
        else:
            t_ij_au = t_matrix[p_idx, r_idx] * scaling_factor

        t_ij_array.append(t_ij_au)
            
        # WKB equivalent irreversible rate formula: Gamma = 2*pi * |t|^2 / (hbar^2 * omega_eff)
        rate_ij_au = (2 * np.pi * t_ij_au**2) / (const.hbar_au**2 * omega_eff_au)
        rate_ij_array.append(rate_ij_au)

        # If product has a higher energy, use the Boltzmann distribution to multiply the rate
        if delta_E_au > 0:
            thermal_penalty = np.exp(-delta_E_au / (const.k_B_au * temperature))
            rate_ij_au = rate_ij_au * thermal_penalty
            thermal_penalties.append(thermal_penalty)
        else:
            thermal_penalties.append(1.0)


        rate_ij_s = rate_ij_au * au_to_inv_s # Convert to s^-1
        
        total_rate_au += rate_ij_au
        if verbose:
            print(f"Channel 0 -> {p_idx}: t = {t_ij_au:.4e} a.u. | Rate = {rate_ij_s:.4e} s^-1")
        
    total_rate_s = total_rate_au * au_to_inv_s
    half_life_s = np.log(2) / total_rate_s if total_rate_s > 0 else float('inf')

    if verbose:
        print("-" * 50)
        print(f"Total tunneling rate: {total_rate_s:.4e} s^-1")
        print(f"Half-life: {half_life_s:.4e} s")

    return tunnel_results.MatrixResult.from_analytic(
        results,
        temperature=temperature,
        reactant_well=reactant_well,
        starting_well=starting_well,
        theta_threshold=theta_threshold,
        harm_attempt_freq=harm_attempt_freq,
        omega_eff_s=omega_eff_au * au_to_inv_s,
        channel_index=np.array(channel_index),
        t_ij_au=np.array(t_ij_array),
        rate_ij_s=np.array(rate_ij_array),
        delta_E_au=np.array(delta_E_array),
        thermal_penalty=np.array(thermal_penalties),
        total_rate_s=total_rate_s,
        half_life_s=half_life_s,
    )
    
    # return total_rate_s, half_life_s











# class AnalyticSolverMixin:


    # # TODO: decrease the number of parameters. They should come as much as possible from class variables
    # def calc_symmetric_energy_levels(self,
    #                                 red_mass=1.0,
    #                                 n_iter=100, 
    #                                 m_step=10.0, 
    #                                 n_levels=8,
    #                                 prec_digits=60,
    #                                 mi=1.0, 
    #                                 hi=1.0, 
    #                                 bi=-41.0, 
    #                                 Li=0.8e-10,
    #                                 matrix_size=1000,
    #                                 raw_energies=False,
    #                                 find_well=False,
    #                                 use_mpmath=False,
    #                                 kinetic_rate=None,
    #                                 verbose=False,
    #                                 verbose_L_fitting=False):

    #     if kinetic_rate is not None:
    #         self.kinetic_rate = kinetic_rate

    #     red_mass = red_mass * const.amu_to_au


    #     react_path_m = self.reaction_path 

    #     if raw_energies:
    #         energies = self.energies
    #         print("Using nonprojected energies, mind you...")
    #     else:
    #         energies = self.proj_energies 
        
    #     # Zeroing in the barrier top
    #     energies = energies - np.max(energies)

    #     # Dynamically set initial L in mass-weighted Bohr if less then the buffer
    #     min_required_L = np.max(np.abs(react_path_m)) * 1.2 # 20% buffer. Should get h to around 10. Small h has some issues.
    #     if Li < min_required_L:
    #         if verbose:
    #             print(f"Warning: Initial L ({Li:.3e}) is smaller than path extent ({min_required_L:.3e}). Increasing L to minimum requirement.")
    #         Li = min_required_L

    #     m_vec, h_vec, b_vec, L_vec, res = [], [], [], [], []
    #     m, h, b, L = mi, hi, bi, Li


    #     # TODO: shouldn't use range(0, n_iter). The fitting should be done increasing m until the difference between
    #     # subsequent splittings is smaller than a threshold, or when it increases or changes too much.
    #     # Calculating the tunneling fitting for multiple m values
    #     old_split = 100
    #     spinner = ["|", "/", "-", "\\"]
    #     for i in range(0,n_iter):
    #         if not verbose_L_fitting and verbose:
    #             print("\rCalculating... " + spinner[i % 4], end="", flush=True)
    #         # old_m, old_h, old_b, old_L = m, h, b, L
    #         if find_well:
    #             p1 = np.array([react_path_m[0], energies[0]])
    #             hTarget = m**2 - 0.25
    #             h, b, L = self._calculate_potential_parameters_stable(p1[0], p1[1], hTarget, red_mass)
    #         else:
    #             h, b, L = self.IRC_L_fit(react_path_m, energies, m, h, b, L, red_mass)

    #         raw_lambdas = self.solve_oblate_spectral(-b, 
    #                                                  m, 
    #                                                  matrix_size=matrix_size, 
    #                                                  use_mpmath=use_mpmath, 
    #                                                  n_levels=n_levels,
    #                                                  prec_digits=prec_digits)



    #         epsilon_level = np.array(raw_lambdas) + 0.5 - (m**2)

    #         # convert factor (8*red_mass*amu*L**2/(hbar**2*np.pi**2)*h_plank)
    #         c_factor = (const.hbar_au**2*np.pi**2) / (8*red_mass*L**2)
    #         split = abs(epsilon_level[0] - epsilon_level[1]) * c_factor * const.hartree_to_cm
    #         diff_split = split - old_split
            
    #         if verbose_L_fitting:
    #             print("m = {:d}, h = {:.4e}, b = {:.3e}, L = {:.5e}, split(cm-1) = {:.5e}".format(int(m), 
    #                                                                                                   h, 
    #                                                                                                   b, 
    #                                                                                                   L, 
    #                                                                                                   float(split)))


    #         m_vec.append(m)
    #         h_vec.append(h)
    #         b_vec.append(b)
    #         L_vec.append(L)
    #         res.append(epsilon_level)
                 
    #         m = m + m_step

    #         old_split = split

    #     if verbose:
    #         print('')
    #     self.m_vec = np.array(m_vec)
    #     self.h_vec = np.array(h_vec)
    #     self.b_vec = np.array(b_vec)
    #     self.L_vec = np.array(L_vec)
    #     self.aps_energy_levels = np.array(res)


    #     # Create a column vector of L^2 for broadcasting: shape (n_iter, 1)
    #     L_squared = self.L_vec[:, np.newaxis]**2
        
    #     scaling_factors = (const.hbar_au**2 * np.pi**2) / (8 * red_mass * L_squared)
        
    #     # Broadcast multiplication: (n_iter, n_levels) * (n_iter, 1)
    #     self.energy_levels = self.aps_energy_levels * scaling_factors

        

    #     freq_s = 0.0
    #     splitting = 0.0

    #     if len(self.energy_levels) > 0 and len(self.energy_levels[-1]) >= 2:

    #         splitting = self.energy_levels[-1][1] - self.energy_levels[-1][0]

    #         # Frequency in s^-1 = DeltaE / h_planck
    #         # DeltaE = hartree , conver to J then divide by h_planck
    #         freq_s = (splitting * const.hartree_to_J / const.h_plank) # Spectroscopic (full period)

    #         if kinetic_rate:
    #             self.tunneling_frequency = 2 * freq_s
    #         else:
    #             self.tunneling_frequency = freq_s

    #     if verbose and len(self.energy_levels) > 0:
    #         coherences = self.calculate_symmetric_level_coherences(n_levels=n_levels)


    #         h_final = self.h_vec[-1]
    #         b_final = self.b_vec[-1]
            
    #         # find the minimum of the dimensionless potential: h*tan^2(y) + b*sin^2(y)
    #         # scan a grid to find the minimum value
    #         # It's not as precise, but it's ok for only printing
    #         y_scan = np.linspace(-np.pi/2 + 1e-3, np.pi/2 - 1e-3, 2000)
    #         u_scan = h_final * (np.tan(y_scan)**2) + b_final * (np.sin(y_scan)**2)
    #         u_min_dim = np.min(u_scan)# This will be negative (relative to barrier top)


    #         final_scaling_factor = scaling_factors[-1][0]
    #         well_depth_h = u_min_dim * final_scaling_factor

    #         print("-" * 90)
    #         print(f"{'Level':<5} | {'Energy (J)':<15} | {'Energy (cm^-1)':<15} | {'Leakage Prob (%)':<16} | {'Theta (deg)':<10}")
    #         print("-" * 90)
    #         for i, (eh, ec) in enumerate(zip(self.energy_levels[-1], self.energy_levels[-1]*const.hartree_to_cm)):
    #             if i >= n_levels: # print only the levels the user want
    #                 break

    #             # This sets the bottom of the well to 0.
    #             e_shifted_h = eh - well_depth_h


    #             e_shifted_cm = e_shifted_h * const.hartree_to_cm
    #             e_shifted_J = e_shifted_h * const.hartree_to_J
                

    #             leak = coherences[i]['leakage_prob'] * 100
    #             theta = coherences[i]['theta_deg'] 

    #             print(f"{i:<5} | {float(e_shifted_J):>15.4e} | {float(e_shifted_cm):>15.2f} | {leak:>16.4f} | {theta:>10.4f}")
    #         print("-" * 90)
    #         if verbose_L_fitting:
    #             print(f"Parameters final values:")
    #             print(f"m = {int(self.m_vec[-1]):d}, h = {self.h_vec[-1]:.4e}, b = {self.b_vec[-1]:.3e}, L = {self.L_vec[-1]:.5e}")
    #         print(f"Ground state splitting (J): {float(splitting*const.hartree_to_J):.4e} J")
    #         print(f"Ground state splitting (cm-1): {float(splitting*const.hartree_to_cm):.4e} cm-1")
    #         # print(f"Ground state splitting: {float(freq_s/const.speed_light):.4e} cm-1")
    #         print(f"Associated Frequency:   {float(freq_s):.4e} s^-1")
















    # def IRC_L_fit(self, X, energies, m_target, hi, bi, Li, red_mass, Lfit_N_iter=1000, Lfit_tol=1e-4, verbose=False):
    #     # the value for h we need to reach
    #     h_target = m_target**2 - 1/4
        
    #     udist, uenerg = self._unitless_var_calc(X, energies, Li, red_mass) # transforming to dimentionless units
    #     param, _ = curve_fit(_aps_sym_potential, udist, uenerg, [hi, bi], ftol=1e-15, xtol=1e-15) #extracting h b
    #     h, b = param # setting inital parameters h and b
        
    #     dh = float(h_target - h) # 0.75 - 0.54 = 0.18
    #     if abs(dh) < Lfit_tol: # First test, if the initial guess is already good, we return it
    #         return h, b, Li 
        
    #     step = ( Li / 10 ) * np.sign(dh) # if the test fails, we set a step for L
    #     L = Li + step # and modify L with the step

    #     for _ in range(0,Lfit_N_iter): # Try Lfit_N_iter times
    #         old_step = np.sign(step) 
            
    #         udist, uenerg = self._unitless_var_calc(X, energies, L, red_mass) # Transform to dimensionless units with the new L
    #         param, _ = curve_fit(_aps_sym_potential, udist, uenerg, [h, b], ftol=1e-15, xtol=1e-15)
    #         h, b = param # New h and b parameters
    #         dh = float(h_target - h)
    #         step = abs(step) * np.sign(dh)

    #         if abs(dh) < Lfit_tol: # test if the threshold was reached
    #             return h, b, L #, result

    #         if old_step != np.sign(step): # Check if we went past the h_target
    #             step = step / 3

    #         L = L + step # update L
    #     if verbose:                 # Final check for physical validity
    #         if h < 0.73: 
    #             print(f"The fitting did not converge properly, try another combination of initial values for h, b and L. h = {h}")
    #             raise ValueError('The fitting did not converge properly.')
    #         print(f"Tolerance was not reached with {Lfit_N_iter} steps. Final result: dh = {dh:.5f} and h = {h} h_target = {h_target}")
    #     return h, b, L







    # # It is faster, but not as reliable. My og function can find L from almost any value. This one has to be within 40% of
    # # the correct value. Leaving it here for perhaps future reference.
    # def IRC_L_fit_alt(self, X, energies, m_target, hi, bi, Li, red_mass, Lfit_tol=1e-4, max_iter=50):
    #     h_target = m_target**2 - 0.25
        
    #     L = Li
    #     h, b = hi, bi
        
    #     # Define a limit for how much L can change in one step (to keep curve_fit stable)
    #     # 5% change maximum per iteration should prevent bad fits
    #     max_step_pct = 0.05 

    #     for i in range(max_iter):

    #         udist, uenerg = self._unitless_var_calc(X, energies, L, red_mass)
    #         try:

    #             popt, _ = curve_fit(_aps_sym_potential, udist, uenerg, p0=[h, b], ftol=1e-15, xtol=1e-15)
    #             h, b = popt
    #         except RuntimeError:
    #             print(f"Fit failed at L={L}")
    #             break 

    #         diff = h - h_target
            
    #         # Check convergence
    #         if abs(diff) < Lfit_tol:
    #             # check if convergence is going anywhere. WHen there is a problem h tends to 0
    #             if h < 0.73:
    #                 print(f"Warning: Converged to h={h:.4f} < 0.73. Result is probably unphysical.")
    #             return h, b, L

    #         # Calculate Gradient (Numerical Derivative) to find smart direction
    #         # We perturb L slightly (0.1%) to see how h changes
    #         dL_probe = L * 0.001 
    #         L_probe = L + dL_probe
            
    #         udist_p, uenerg_p = self._unitless_var_calc(X, energies, L_probe, red_mass)
    #         try:
    #             popt_p, _ = curve_fit(_aps_sym_potential, udist_p, uenerg_p, p0=[h, b], ftol=1e-15, xtol=1e-15)
    #             h_probe, _ = popt_p
                
    #             # Slope dh/dL
    #             slope = (h_probe - h) / dL_probe
    #         except:
    #             slope = 0 # Fallback to avoid crash

    #         # Newton Step: delta_L = - error / slope
    #         if abs(slope) > 1e-12:
    #             step = - diff / slope
    #         else:
    #             # If slope is flat, fallback to simple small step 
    #             step = L * 0.01 * np.sign(-diff)

    #         # DAMPING / CLAMPING 
    #         # Do not allow L to change by more than max_step_pct
    #         # This keeps L inside the "basin of attraction" for curve_fit
    #         max_step = L * max_step_pct
            
    #         if abs(step) > max_step:
    #             step = max_step * np.sign(step)

    #         # Safety check for minimum L (cannot be smaller than max(X))
    #         min_L = np.max(np.abs(X)) * 1.01
    #         if (L + step) < min_L:
    #             step = (min_L - L) / 2 # move halfway to the limit

    #         L = L + step
            
    #     print(f"Tolerance not reached in {max_iter} steps. Final diff: {diff}")
    #     return h, b, L







    # def _calculate_potential_parameters_stable(self, x_phys, E_phys, h_target, mu):
    #     """Calculates L and the dimensionless potential parameters h and b.

    #     This function acts as a wrapper that uses a numerically stable method
    #     to find the length scale L that maps a given physical minimum (x_phys, E_phys)
    #     to a dimensionless double-well potential characterized by a target h-parameter.
    #     It returns L along with the self-consistent h and b parameters.

    #     Args:
    #         x_phys (float): The physical x-coordinate of the minimum (in bohrs).
    #         E_phys (float): The physical energy of the minimum (in hartree).
    #         h_target (float): The target h-parameter for the dimensionless potential.
    #         mu (float): The effective mass of the particle (in atomic mass units).

    #     Returns:
    #         list(float) | None: A list containing the calculated 'L', 'h',
    #                                 and 'b' parameters. The returned 'h' is recalculated
    #                                 from the found L for consistency, and it should match
    #                                 the h_target to a high precision. Returns None if
    #                                 the solver fails.
    #     """
    #     tau_found, L_found = self._find_tau_and_L_stable(x_phys, E_phys, h_target, mu)

    #     if L_found is None:
    #         print("Could not determine a valid solution. Aborting.")
    #         return None
        
    #     # Recalculate epsilon using the found L to check consistency
    #     epsilon = (8 * mu * L_found**2 * E_phys) / (const.hbar_au**2 * np.pi**2)

    #     # Recalculate h from the results
    #     h_recalculated = -epsilon / (np.tan(tau_found)**4)
    #     b_calculated = -h_recalculated / (np.cos(tau_found)**4)
        
    #     return h_recalculated, b_calculated, L_found








    # def _find_tau_and_L_stable(self, x_phys, E_phys, h_target, mu):
    #     """Solves for the dimensionless coordinate tau and the length scale L.

    #     This function sets up a numerically stable objective function in terms of tau (τ)
    #     and solves for its root in the physical domain (0, pi/2). It then uses the
    #     found tau to calculate the corresponding length scale L.

    #     Args:
    #         x_phys (float): The physical x-coordinate of the minimum (in meters).
    #         E_phys (float): The physical energy of the minimum (in Joules).
    #         h_target (float): The target h-parameter for the dimensionless potential,
    #                         used to set up the equation.
    #         mu (float): The effective mass of the particle (in kg).

    #     Returns:
    #         tuple[float | None, float | None]: A tuple containing the found tau and L.
    #                                         Returns (None, None) if the solver fails.
    #     """

    #     # define the stable objective function in terms of tau
    #     def objective_function_tau(tau):
    #         # physical domain for tau is (0, pi/2)
    #         if tau <= 0 or tau >= np.pi / 2:
    #             return np.inf
    #         term1 = (2 * mu * x_phys**2 * E_phys) / (const.hbar_au**2 * tau**2)
    #         term2 = h_target * np.tan(tau)**4
    #         return term1 + term2

    #     # bracket for tau is always (0, pi/2)
    #     # use a small epsilon to avoid division by zero at tau=0
    #     lower_bound_tau = 1e-9
    #     upper_bound_tau = np.pi/2 - 1e-9

    #     try:
    #         sol = root_scalar(
    #             objective_function_tau,
    #             method='brentq',
    #             bracket=[lower_bound_tau, upper_bound_tau]
    #         )
    #         if sol.converged:
    #             tau_found = sol.root
    #             # calculate the corresponding L
    #             L_found = (np.pi * x_phys) / (2 * tau_found)
    #             return tau_found, L_found
    #         else:
    #             return None, None
    #     except ValueError:        # This can happen if the function doesn't cross zero in the bracket
    #         return None, None








    # def solve_oblate_spectral(self, b, m_val, n_levels=2, matrix_size=500, use_mpmath=True, prec_digits=60):
    #     """
    #     Solves the Oblate Spheroidal eigenvalues.
        
    #     Args:
    #         n_levels (int): The number of energy levels to return (e.g., 2, 4, 10).
    #         use_mpmath (bool): If True, uses arbitrary precision arithmetic to resolve tiny splittings.
    #         prec_digits (int): Number of decimal digits of precision (try 60 or 100).
    #     """
    #     if not use_mpmath:
    #         # numpy implementation, it's fast but can't get very small splittings (looking at you trimethylamine)
    #         c_sq = b
    #         m_val = float(m_val)
            
    #         def get_vals_numpy(start_index):
    #             N = matrix_size
    #             js = np.arange(N)
    #             n_degrees = m_val + start_index + 2 * js
                
    #             term_c = (2*n_degrees*(n_degrees+1) - 2*m_val**2 - 1) / ((2*n_degrees - 1)*(2*n_degrees + 3))
    #             diag = n_degrees * (n_degrees + 1) - c_sq * term_c 
                
    #             n_sub = n_degrees[:-1]
    #             num = (n_sub + 2 + m_val) * (n_sub + 2 - m_val) * (n_sub + 1 + m_val) * (n_sub + 1 - m_val)
    #             den = (2*n_sub + 1) * (2*n_sub + 5) * (2*n_sub + 3)**2
                
    #             safe_array = np.maximum(0, num / den) # [FIXED] Clamp small negatives to 0, then take sqrt
    #             off_diag = -c_sq * np.sqrt(safe_array)
                
    #             vals, _ = linalg.eigh_tridiagonal(diag, off_diag)
    #             return vals

    #         # get all eigenvalues for even and odd blocks
    #         vals_even = get_vals_numpy(0)
    #         vals_odd = get_vals_numpy(1)
            
    #         # interleave them: Even0, Odd0, Even1, Odd1...
    #         combined = []
    #         for k in range(n_levels):
    #             if k % 2 == 0:
    #                 combined.append(vals_even[k // 2])
    #             else:
    #                 combined.append(vals_odd[k // 2])
    #         return combined

    #     else:
    #         # mpmath implementation, huge number of precision digits
    #         mp.mp.dps = prec_digits
            
    #         m_mp = mp.mpf(m_val)
    #         c_sq = mp.mpf(b)
            
    #         # helper to generate matrix elements for a specific block (Even=0 / Odd=1)
    #         def get_matrix_elements(start_index):
    #             diag = []
    #             off_diag_sq = [] # Store squared off-diagonal for Sturm calculation
                
    #             for j in range(matrix_size):
    #                 n = m_mp + start_index + 2*j
                    
    #                 # diagonal
    #                 term_c = (2*n*(n+1) - 2*m_mp**2 - 1) / ((2*n - 1)*(2*n + 3))
    #                 d_val = n*(n+1) - c_sq * term_c
    #                 diag.append(d_val)
                    
    #                 # off-diagonal (squared)
    #                 if j < matrix_size - 1:
    #                     n_sub = n
    #                     num = (n_sub + 2 + m_mp) * (n_sub + 2 - m_mp) * (n_sub + 1 + m_mp) * (n_sub + 1 - m_mp)
    #                     den = (2*n_sub + 1) * (2*n_sub + 5) * (2*n_sub + 3)**2
    #                     # We only need the square for Sturm sequence, which avoids one sqrt()
    #                     od_sq = (c_sq**2) * (num / den) 
    #                     off_diag_sq.append(od_sq)
    #             return diag, off_diag_sq

    #         # Sturm Sequence: Count eigenvalues smaller than sigma
    #         def count_below(sigma, diag, off_diag_sq):
    #             count = 0
    #             q = diag[0] - sigma
    #             if q < 0: count += 1
                
    #             for i in range(1, len(diag)):
    #                 if q == 0: q = mp.mpf('1e-50')
    #                 # q_i = (d_i - sigma) - e_{i-1}^2 / q_{i-1}
    #                 q = (diag[i] - sigma) - off_diag_sq[i-1]/q
    #                 if q < 0: count += 1
    #             return count

    #         # find the k-th eigenvalue using Bisection
    #         def find_eigenvalue_k(k, diag, off_diag_sq):
    #             # Gerschgorin bounds estimate
    #             # max off-diagonal estimate (approx sqrt of max sq)
    #             max_off = mp.sqrt(max(off_diag_sq)) if off_diag_sq else 0
    #             low = min(diag) - max_off - 1
    #             high = max(diag) + max_off + 1
                
    #             # Bisection
    #             # we stop when the interval size is smaller than 10^-dps
    #             # or a fixed number of iterations
    #             for _ in range(250):
    #                 mid = (low + high) / 2
    #                 if count_below(mid, diag, off_diag_sq) <= k:
    #                     low = mid
    #                 else:
    #                     high = mid
    #             return (low + high) / 2

    #         # Pre-calculate matrix elements for both blocks
    #         diag_even, off_sq_even = get_matrix_elements(0)
    #         diag_odd, off_sq_odd = get_matrix_elements(1)
            
    #         combined_results = []
            
    #         # request specific levels
    #         # physical level 0 -> Even block, index 0
    #         # physical level 1 -> Odd block, index 0
    #         # physical level 2 -> Even block, index 1
    #         # ...
            
    #         for i in range(n_levels):
    #             target_index = i // 2
                
    #             if i % 2 == 0:
    #                 # even Parity
    #                 val = find_eigenvalue_k(target_index, diag_even, off_sq_even)
    #             else:
    #                 # odd Parity
    #                 val = find_eigenvalue_k(target_index, diag_odd, off_sq_odd)
                
    #             combined_results.append(val)
                
    #         return combined_results








    # def solve_asymmetric_finite_difference(self, h, b, a, N=5000, n_levels=6, epsilon=1e-3):
    #     """
    #     Solves the Asymmetric Double Well using a Direct Tridiagonal Solver.
    #     Includes numerical stabilization for singular boundaries.
    #     """
    #     # grid, domain is (-pi/2, pi/2). 
    #     L_bound = np.pi / 2.0
        
    #     # [FIX] Increased epsilon from 1e-6 to 1e-3. avoid large values.
    #     # Changed to a paramneter to allow the user to change it.
    #     # epsilon = 1e-3 
        
    #     y = np.linspace(-L_bound + epsilon, L_bound - epsilon, N)
    #     dy = y[1] - y[0]
        
        
    #     # U = h tan^2 + sqrt(h) sin/cos^2 - b sin^2 + a sin
    #     c = np.sqrt(h) 
        
    #     # [TODO] Changed to method. Check results and remove it later.
    #     # tan_y = np.tan(y)
    #     # sin_y = np.sin(y)
    #     # cos_sq = np.cos(y)**2
    #     # U = (h * tan_y**2) + (c * sin_y / cos_sq) - (b * sin_y**2) + (a * sin_y)

    #     U = _aps_asym_potential(y, h, b, a)
        
    #     # [FIX] Clamp the potential.
    #     # This prevents floating point issues if the grid accidentally hits a singularity.
    #     # 1e8 should high enough to be an "infinite wall" for the physics, but low enough for math.
    #     U = np.clip(U, -1e8, 1e8)

    #     # construct tridiagonal matrix elements
    #     # the Hamiltonian is H = -d^2/dy^2 + U
    #     kin_coeff = 1.0 / (dy**2)
        
    #     d = 2 * kin_coeff + U      # main diagonal elements
    #     e = -kin_coeff * np.ones(N-1) # off-diagonal elements
        
    #     # diagonilize and get n_levels
    #     eigenvalues, eigenvectors = linalg.eigh_tridiagonal(
    #         d, e, select='i', select_range=(0, n_levels-1)
    #     )
        
    #     # notmalize
    #     for i in range(eigenvalues.shape[0]):
    #         norm = np.sqrt(trapz(eigenvectors[:, i]**2, x=y))
    #         eigenvectors[:, i] /= norm
            
    #     return eigenvalues, eigenvectors, y








    # def calc_asymmetric_energy_levels(self, 
    #                                   hi=50.0, 
    #                                   bi=150.0, 
    #                                   ai=0.0, 
    #                                   Li=1, #0.9e-10, 
    #                                   n_iter=100, 
    #                                   L_step=0.2,#0.02e-10, 
    #                                   red_mass=1.0,
    #                                   raw_energies=False,
    #                                   find_well=False,
    #                                   n_levels = 6,
    #                                   kinetic_rate=None,
    #                                   verbose=True,
    #                                   verbose_L_fitting=False):
    #     """
    #     Wrapper to solve the asymmetric potential and convert results to physical units.
    #     """

    #     if kinetic_rate is not None:
    #         self.kinetic_rate = kinetic_rate

    #     # Used to convert the displacements from bohrs to meters
    #     # Since the view won't change anything, maybe use the class variable?
    #     react_path_m = self.reaction_path 

    #     red_mass = red_mass * const.amu_to_au

    #     if raw_energies:
    #         energies = self.energies 
    #         print("Using nonprojected energies...")
    #     else:
    #         energies = self.proj_energies 
        
    #     # zeroing
    #     energies = energies - np.max(energies)

    #     h_vec = []
    #     b_vec = []
    #     a_vec = []
    #     L_vec = []
    #     res = []
    #     vectors = []
        
    #     h, b, a, L = hi, bi, ai, Li        


    #     min_required_L = np.max(np.abs(react_path_m)) * 1.2 # 20% buffer. Should get h to around 10. Small h has some issues.
    #     if L < min_required_L:
    #         if verbose:
    #             print(f"Warning: Initial L ({L:.3e}) is smaller than path extent ({min_required_L:.3e}). Increasing L to minimum requirement.")
    #         L = min_required_L

    #     for i in range(n_iter):

    #         udist, uenergies = self._unitless_var_calc(react_path_m, energies, L, red_mass)
    #         c0 = [h, b, a]
    #         try:
    #             param, cov = curve_fit(_aps_asym_potential, udist, uenergies, c0, ftol=1e-15, xtol=1e-15)
    #         except RuntimeError:
    #             print(f"Fit failed at iteration {i} with L={L}")
    #             break

    #         h_fit, b_fit, a_fit = param

    #         if h_fit < 0.73:
    #             if verbose:
    #                 print(f"Warning: Fitted h={h_fit:.4f} < 0.73 at iter {i}. Check initial values.")
    #                 raise ValueError('Fitted h is too small!')

    #         eig_vals, eig_vecs, grid = self.solve_asymmetric_finite_difference(h_fit, b_fit, a_fit, n_levels=n_levels)

    #         splitting_mp = abs(eig_vals[0] - eig_vals[1])

    #         split = splitting_mp / (8*red_mass*L**2/(const.hbar_au**2*np.pi**2)) * const.hartree_to_J / const.h_plank
            
    #         if verbose_L_fitting:
    #             print("h = {:.4e}, b = {:.4e}, a = {:.4e}, L = {:.5e}, tunneling frequency = {:.5e}".format(h_fit, 
    #                                                                                                   b_fit, 
    #                                                                                                   a_fit, 
    #                                                                                                   L, 
    #                                                                                                   float(split)))

    #         h_vec.append(h_fit)
    #         b_vec.append(b_fit)
    #         a_vec.append(a_fit)
    #         L_vec.append(L)
    #         res.append(eig_vals)
    #         vectors.append(eig_vecs)

    #         h, b, a = h_fit, b_fit, a_fit
    #         L += L_step

    #     self.h_vec = np.array(h_vec)
    #     self.b_vec = np.array(b_vec)
    #     self.a_vec = np.array(a_vec) 
    #     self.L_vec = np.array(L_vec)

    #     # energy levels in dimensionless units
    #     # shape: (n_iter, n_levels)
    #     self.che_energy_levels = np.array(res)

    #     # Create a column vector of L^2 for broadcasting: shape (n_iter, 1)
    #     L_squared = self.L_vec[:, np.newaxis]**2
        
    #     scaling_factors = (const.hbar_au**2 * np.pi**2) / (8 * red_mass * L_squared)
        
    #     energies_hartree = self.che_energy_levels * scaling_factors
    #     energies_joules = energies_hartree * const.hartree_to_J
        
    #     # hartree
    #     self.energy_levels = energies_hartree

    #     freq_s = 0.0
    #     splitting = 0.0
        
    #     if len(energies_joules) > 0 and len(energies_joules[-1]) >= 2:
    #         e0 = energies_joules[-1][0]
    #         e1 = energies_joules[-1][1]
    #         splitting = e1 - e0
            
    #         # Frequency in s^-1 = DeltaE / h_planck
    #         freq_s = (splitting / const.h_plank) # Spectroscopic (full period)

    #         if kinetic_rate:
    #             self.tunneling_frequency = 2 * freq_s
    #         else:
    #             self.tunneling_frequency = freq_s


    #     # Only calculate the coherences if they are asked by the user
    #     self.coherences = self.calculate_level_coherences(n_levels=len(energies_joules[-1])) 

    #     if verbose and len(energies_joules) > 0:
            
    #         print("-" * 75)
    #         print(f"{'Level':<5} | {'Energy (J)':<15} | {'Energy (cm^-1)':<15} | {'Leakage Prob (%)':<16} | {'Theta (deg)':<10}")
    #         print("-" * 75)
    #         for i, (ej, ec) in enumerate(zip(energies_joules[-1], energies_hartree[-1]*const.hartree_to_cm)):
                
    #             leak = self.coherences[i]['leakage_prob']
    #             theta = self.coherences[i]['theta_deg']
                
    #             print(f"{i:<5} | {ej:<15.4e} | {ec:<15.2f} | {leak*100:<16.2f} | {theta:<10.4f}")

    #         print("-" * 75)
            
    #         print(f"Ground state splitting: {splitting:.4e} J")
    #         print(f"Ground state splitting: {freq_s/const.speed_light:.4e} cm-1")
    #         print(f"Associated Frequency:   {freq_s:.4e} s^-1")
            
    #         if self.a_vec[-1] != 0:
    #             print("Note: In asymmetric wells, splitting includes the asymmetry offset (detuning).")








    # def calculate_symmetric_level_coherences(self, n_levels=None):
    #     """
    #     Calculates the coherence (mixing angle and leakage probability) 
    #     for the symmetric energy levels using a finite-difference grid.
    #     """
    #     if not hasattr(self, 'h_vec') or len(self.h_vec) == 0:
    #         return []

    #     h = self.h_vec[-1]
    #     b = self.b_vec[-1]
        
    #     if n_levels is None:
    #         n_levels = len(self.energy_levels[-1])

    #     # grid
    #     N_grid = 2000
    #     epsilon = 1e-3
    #     y_grid = np.linspace(-np.pi/2 + epsilon, np.pi/2 - epsilon, N_grid)
    #     dy = y_grid[1] - y_grid[0]
        
    #     # [FIXED] reuse method symmetric potential. Remove after testing
    #     # U_dim = h * (np.tan(y_grid)**2) + b * (np.sin(y_grid)**2)
    #     U_dim = _aps_sym_potential(y_grid,h, b)
    #     U_dim = np.clip(U_dim, -1e9, 1e9)
        
    #     kin_coeff = 1.0 / (dy**2)
    #     diag = 2 * kin_coeff + U_dim
    #     off_diag = -kin_coeff * np.ones(N_grid - 1)
        
    #     evals_dim, evecs = linalg.eigh_tridiagonal(
    #         diag, off_diag, select='i', select_range=(0, n_levels-1)
    #     )
        
    #     # symmetric DWP barrier is in 0.0 by definition
    #     barrier_idx = np.argmin(np.abs(y_grid - 0.0))

    #     coherences = []
    #     for i in range(len(evals_dim)):
    #         # normalize wasvefunction
    #         psi = evecs[:, i]
    #         norm = np.sqrt(trapz(y=psi**2, x=y_grid))
    #         psi = psi / norm
            
    #         prob_density = psi**2
            
    #         prob_left = trapz(y=prob_density[:barrier_idx], x=y_grid[:barrier_idx])
    #         prob_right = trapz(y=prob_density[barrier_idx:], x=y_grid[barrier_idx:])
            
    #         leakage_prob = min(prob_left, prob_right)
    #         theta_rad = np.arcsin(np.sqrt(max(0, leakage_prob)))
            
    #         coherences.append({
    #             'level': i,
    #             'leakage_prob': leakage_prob,
    #             'theta_deg': np.degrees(theta_rad)
    #         })
            
    #     return coherences








    # # [TODO] change the name to add asymmetric to it...
    # def calculate_level_coherences(self, n_levels=None):
    #     """
    #     Calculates the coherence (mixing angle and leakage probability) 
    #     for the calculated asymmetric energy levels.
    #     """
    #     if not hasattr(self, 'h_vec') or len(self.h_vec) == 0:
    #         print("No calculation results found. Run calc_asymmetric_energy_levels first.")
    #         return []

    #     h = self.h_vec[-1]
    #     b = self.b_vec[-1]
    #     a = self.a_vec[-1]
        
    #     if n_levels is None:
    #         n_levels = len(self.energy_levels[-1])

    #     # Recalculate grid and wave functions for the final parameters
    #     eig_vals, eig_vecs, y_grid = self.solve_asymmetric_finite_difference(h, b, a, n_levels=n_levels)

    #     # potential (careful on the sign)
    #     # [FIXED] reuse method symmetric potential. Remove after testing
    #     # tan_y = np.tan(y_grid)
    #     # sin_y = np.sin(y_grid)
    #     # cos_sq = np.cos(y_grid)**2
    #     # U_dim = (h * tan_y**2) + (c * sin_y / cos_sq) - (b * sin_y**2) + (a * sin_y)
    #     # c = np.sqrt(h)
    #     U_dim = _aps_asym_potential(y_grid, h, b, a)
        
    #     edge_buffer = len(U_dim) // 10  # exclude diverging edges

    #     # Find both wells (minima) in left and right halves of the interior
    #     interior_start = edge_buffer
    #     interior_end = len(U_dim) - edge_buffer
    #     midpoint = len(U_dim) // 2

    #     left_min_idx = interior_start + np.argmin(U_dim[interior_start:midpoint])
    #     right_min_idx = midpoint + np.argmin(U_dim[midpoint:interior_end])

    #     # Barrier is the maximum between the two wells
    #     barrier_idx = left_min_idx + np.argmax(U_dim[left_min_idx:right_min_idx])

    #     coherences = []
    #     for i in range(n_levels):
    #         psi = eig_vecs[:, i]    
    #         # Normalize on the actual grid (continuous normalization)
    #         norm = np.sqrt(trapz(y=psi**2, x=y_grid))
    #         psi = psi / norm
            
    #         prob_density = psi**2
            
    #         prob_left = trapz(y=prob_density[:barrier_idx], x=y_grid[:barrier_idx])
    #         prob_right = trapz(y=prob_density[barrier_idx:], x=y_grid[barrier_idx:])
            
    #         # leakage is the smaller one. Could check a better way of defining these.
    #         leakage_prob = min(prob_left, prob_right)
            
    #         # mixing angle theta = arcsin(sqrt(leakage_prob))
    #         # max(0, val) avoid negative float point error
    #         theta_rad = np.arcsin(np.sqrt(max(0, leakage_prob)))
    #         theta_deg = np.degrees(theta_rad)
            
    #         coherences.append({
    #             'level': i,
    #             'prob_left': prob_left,
    #             'prob_right': prob_right,
    #             'leakage_prob': leakage_prob,
    #             'theta_deg': theta_deg
    #         })
            
    #     self.coherences = coherences
    #     return coherences








    # def _generate_isolated_potentials(self, U, y):
    #     """
    #     Takes the full double well potential and splits it into 
    #     isolated left and right wells by replacing the opposite side 
    #     with a large numerical wall.
    #     """
    #     N = len(U)
    #     edge_buffer = N // 10
    #     midpoint = N // 2

    #     # getting minima indexes instead of the barrier
    #     left_min_idx = edge_buffer + np.argmin(U[edge_buffer:midpoint])
    #     right_min_idx = midpoint + np.argmin(U[midpoint:-edge_buffer])
    #     barrier_idx = left_min_idx + np.argmax(U[left_min_idx:right_min_idx])
        
    #     # making Left well
    #     U_left = np.copy(U)
    #     U_left[barrier_idx:] = 1e8  # Wipe out the right well
        
    #     # making right well
    #     U_right = np.copy(U)
    #     U_right[:barrier_idx] = 1e8 # Wipe out the left well
        
    #     return U_left, U_right, left_min_idx, right_min_idx, barrier_idx








    # def _solve_custom_potential(self, U, dy, n_levels):
    #     """
    #     A modular version of the finite difference solver that accepts a custom U.
    #     """
    #     N = len(U)
    #     kin_coeff = 1.0 / (dy**2)
        
    #     d = 2 * kin_coeff + U      # main diagonal elements
    #     e = -kin_coeff * np.ones(N-1) # off-diagonal elements
        
    #     # Diagonalizing
    #     eigenvalues, eigenvectors = linalg.eigh_tridiagonal(
    #         d, e, select='i', select_range=(0, n_levels-1)
    #     )
        
    #     # Normalizing
    #     for i in range(eigenvalues.shape[0]):
    #         norm = np.sqrt(trapz(eigenvectors[:, i]**2, dx=dy))
    #         eigenvectors[:, i] /= norm
            
    #     return eigenvalues, eigenvectors








    # def _calculate_tunneling_matrix(self, U_full, vecs_L, vecs_R, dy, n_levels):
    #     """
    #     Calculates the off-diagonal tunneling amplitudes t_{ij} between 
    #     localized left states and right states.
    #     """
    #     N = len(U_full)
    #     kin_coeff = 1.0 / (dy**2)
        
    #     # Construct the full Hamiltonian as a sparse matrix
    #     main_diag = 2 * kin_coeff + U_full
    #     off_diag = -kin_coeff * np.ones(N-1)
    #     H_full = diags([off_diag, main_diag, off_diag], [-1, 0, 1])
        
    #     # Initialize the amplitude matrix
    #     t_matrix = np.zeros((n_levels, n_levels))
        
    #     for i in range(n_levels):
    #         for j in range(n_levels):
    #             phi_L_i = vecs_L[:, i]
    #             phi_R_j = vecs_R[:, j]
                
    #             # Apply the full Hamiltonian to the right state: H |phi_R>
    #             H_phi_R = H_full.dot(phi_R_j)
                
    #             # Overlap with the left state: <phi_L | H | phi_R>
    #             t_ij = trapz(phi_L_i * H_phi_R, dx=dy)
                
    #             # Take the absolute value (the phase should be usually arbitrary here)
    #             t_matrix[i, j] = np.abs(t_ij)
                
    #     return t_matrix








    # def calculate_matrix_tunneling_rate(self, 
    #                                     starting_well='higher', # is there an use case for the opposite?
    #                                     theta_threshold=5.0,
    #                                     red_mass=1.0,
    #                                     temperature=3.0,
    #                                     harm_attempt_freq=False,
    #                                     verbose=True):
    #     """
    #     Calculates the irreversible tunneling rate from the higher well ground state 
    #     to all localized states in the global minimum well, assuming a T=temperature matrix bath.
    #     """

    #     # Converting the mass to au
    #     red_mass_au = red_mass * const.amu_to_au

    #     # Run diagnostics to get coherence data and grid
    #     coherences = self.calculate_level_coherences()
        
    #     # Get the final parameters 
    #     h, b, a, L = self.h_vec[-1], self.b_vec[-1], self.a_vec[-1], self.L_vec[-1]
        
    #     # Re-solve to get the exact grid and dy
    #     _, _, y_grid = self.solve_asymmetric_finite_difference(h, b, a)
    #     dy = y_grid[1] - y_grid[0]
        
    #     # Build potential and clip to avoid numerical issues
    #     U_full = _aps_asym_potential(y_grid, h, b, a)
    #     U_full = np.clip(U_full, -1e8, 1e8)
        
    #     # generating the isolated potentials
    #     ( U_L, U_R, left_min_idx, 
    #         right_min_idx, barrier_idx ) = self._generate_isolated_potentials(U_full, y_grid)
        
    #     # Determine which well is the higher one
    #     if U_full[left_min_idx] > U_full[right_min_idx]:
    #         reactant_well = 'left'
    #         reactant_min_idx = left_min_idx
    #     else:
    #         reactant_well = 'right'
    #         reactant_min_idx = right_min_idx
            
    #     # Filter states based on the mixing angle threshold
    #     reactant_states = []
    #     product_states = []
    #     V_min_reactant = U_full[left_min_idx] if reactant_well == 'left' else U_full[right_min_idx]

    #     for coh in coherences:
    #         if coh['theta_deg'] > theta_threshold:
    #             continue # [IMPORTANT] skipping delocalized states 
                
    #         # Classify based on where the probability majority lies
    #         if coh['prob_left'] > coh['prob_right']:
    #             if reactant_well == 'left':
    #                 reactant_states.append(coh)
    #             else:
    #                 product_states.append(coh)
    #         else:
    #             if reactant_well == 'right':
    #                 reactant_states.append(coh)
    #             else:
    #                 product_states.append(coh)
                    
    #     if not reactant_states:
    #         raise ValueError("No localized ground state found in the reactant well!")
            

    #     # Compute the effective attempt frequency (omega_eff) from ZPE
    #     # E_L - V_min = 0.5 * hbar * omega_eff
    #     E_reactant_0 = self.energy_levels[-1][reactant_states[0]['level']] # in Hartree
        
    #     # Convert V_min from dimensionless to Hartree 
    #     scaling_factor = (const.hbar_au**2 * np.pi**2) / (8 * red_mass_au * L**2)

    #     # Map dimensionless variables to physical units
    #     x_phys = (2.0 * L / np.pi) * y_grid
    #     V_phys = U_full * scaling_factor
        
    #     if harm_attempt_freq:
    #         # Harmonic approximation for omega
    #         V_min_hartree = V_min_reactant * scaling_factor
    #         omega_eff_au = 2.0 * (E_reactant_0 - V_min_hartree) / const.hbar_au
    #     else:
    #         # Attempt frequency classically integrated (check)
    #         omega_eff_au = self.calculate_classical_frequency(
    #             x_array=x_phys, 
    #             V_array=V_phys, 
    #             E_target=E_reactant_0, 
    #             mass=red_mass_au, 
    #             min_idx=reactant_min_idx, 
    #             barrier_idx=barrier_idx, 
    #             reactant_well=reactant_well
    #         )
        
    #     # get only the relevant levels to avoid numerical issues
    #     max_level_needed = max([s['level'] for s in (reactant_states + product_states)]) + 1
        
    #     _, vecs_L = self._solve_custom_potential(U_L, dy, n_levels=max_level_needed)
    #     _, vecs_R = self._solve_custom_potential(U_R, dy, n_levels=max_level_needed)
        

    #     # Calculate amplitudes and total rate
    #     t_matrix = self._calculate_tunneling_matrix(U_full, vecs_L, vecs_R, dy, n_levels=max_level_needed)
        
    #     total_rate_au = 0.0
    #     r_idx = reactant_states[0]['level'] # Starting from the reactant ground state
        
    #     au_to_inv_s = const.hartree_to_J / const.hbar

    #     if verbose:
    #         print(f"\n--- Matrix Tunneling Analysis (T={temperature}K) ---")
    #         print(f"Reactant well: {reactant_well.capitalize()}")
    #         print(f"Effective Attempt Frequency: {omega_eff_au * au_to_inv_s:.4e} s^-1")
    #         print("-" * 50)
        
    #     k_B_au = 3.166811563e-6

    #     for p_state in product_states:
    #         p_idx = p_state['level']

    #         # product height in relation to the reactant, to check  viability
    #         E_product = self.energy_levels[-1][p_idx]
    #         delta_E_au = E_product - E_reactant_0
            
    #         # Extract t depending on direction
    #         if reactant_well == 'left':
    #             t_ij_au = t_matrix[r_idx, p_idx] * scaling_factor
    #         else:
    #             t_ij_au = t_matrix[p_idx, r_idx] * scaling_factor
                
    #         # WKB-equivalent irreversible rate formula: Gamma = 2*pi * |t|^2 / (hbar^2 * omega_eff)
    #         rate_ij_au = (2 * np.pi * t_ij_au**2) / (const.hbar_au**2 * omega_eff_au)

    #         # If product has a higher energy, use the Boltzmann distribution to multiply the rate
    #         if delta_E_au > 0:
    #             thermal_penalty = np.exp(-delta_E_au / (k_B_au * temperature))
    #             rate_ij_au = rate_ij_au * thermal_penalty

    #         rate_ij_s = rate_ij_au * au_to_inv_s # Convert to s^-1
            
    #         total_rate_au += rate_ij_au
    #         if verbose:
    #             print(f"Channel 0 -> {p_idx}: t = {t_ij_au:.4e} a.u. | Rate = {rate_ij_s:.4e} s^-1")
            
    #     total_rate_s = total_rate_au * au_to_inv_s
    #     half_life_s = np.log(2) / total_rate_s if total_rate_s > 0 else float('inf')

    #     if verbose:
    #         print("-" * 50)
    #         print(f"Total tunneling rate: {total_rate_s:.4e} s^-1")
    #         print(f"Half-life: {half_life_s:.4e} s")
        
    #     return total_rate_s, half_life_s
