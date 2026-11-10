import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.integrate import quad
from scipy.optimize import brentq
from tunkit.core import constants as const
from tunkit.io import logging 
from tunkit.path import projections 
from tunkit.solvers import tunnel_results
from tunkit.solvers import general









def calculate_relevant_vib(path, 
                           rms_threshold=1e-3, 
                           gradient_source=1,
                           logger=None):
    """
    Calculates the Effective Attack Frequency (nu_eff) using the Hessian of the Reactant
    and the Gradient of the first stable point along the path.

    Algorithm:
    1. Uses the gradient_source point or finds the point closes to the reactant
        that is stable.
    2. Projects the reactant's Hessian onto the gradient at the point.
    3. Identifies the overlap in normal modes.
    4. Selects the highest overlap as the attack frequency and prints report.
    
    Parameters
    ----------
    gradient_source : int
        The point in the MEP to compare the frequencies with the reactant.
        It should be as close as possible to the reactant, so 1.

    rms_threshold : float
        The RMS gradient threshold in Hartree/Bohr, if legacy.
        Default is 1e-3.
    
    Returns
    -------
    float
        The effective frequency in cm^-1.
    """

    # [TODO] not used anymore. remove after test.
    # if gradient_source < 0:
    #     reactant_idx, grad_source_idx = self.find_stable_gradient_source(rms_threshold, verbose)
    # else:

    # lists to output to fill the class on an upper level
    freqs = []
    overlaps = []
    contributions = []

    reactant_idx = 0
    grad_source_idx = gradient_source

    final_rms = np.sqrt(np.mean(path.gradients[grad_source_idx]**2))


    hess_mw = path.mw_hessians[reactant_idx]
    
    grad_mw = path.mw_gradients[grad_source_idx]
    
    # REACTANT (index 0)
    # reactant coords to define TR modes because we are projecting the reactant's hessian.
    coords = path.proj_coordinates[reactant_idx] 

    # project translations and rotations out
    grad_mw_clean = projections._project_out_tr(coords, grad_mw, path.masses)

    norm = np.linalg.norm(grad_mw_clean)

    
    if norm < 1e-12:
            print("-" * 48)
            print("Error: Gradient vector is effectively zero even after selection logic. Choose another point in the MEP!")
            raise ValueError("Gradient is ~zero.")

    logger.log("-" * 48)
    logger.log(f"Attack Frequency Analysis")
    logger.log("-" * 48)
    logger.log(f"{'Mode':<5} | {'Freq (cm-1)':<12} | {'Overlap':<10} | {'Contribution':<10}")
    logger.log("-" * 48)

    u = grad_mw_clean / norm


    best_overlap = -1.0
    winner_freq = 0.0
    

    reactant_freqs = path.path_frequencies[reactant_idx]
    # Shape is (3N, 3N) -> (Coords, Modes)
    reactant_modes = path.normal_modes[reactant_idx]

    m_repeat = np.repeat(path.masses, 3)

    mw_factor = np.sqrt(m_repeat * const.amu_to_au)

    for i in range(len(reactant_freqs)):
        freq = reactant_freqs[i]

        if freq < 10.0:
            continue  # skip projected/imaginary modes [TODO] better logic to imaginary frequencies... print? stop?

        # mode_vec = self.normal_modes[reactant_idx][i]

        mode_vec = reactant_modes[:, i] 
        # mode_vec_mw = mode_vec * mw_factor
        
        mode_norm = np.linalg.norm(mode_vec)
        if mode_norm > 1e-8:
            mode_vec = mode_vec / mode_norm

        # overlap = dot product
        overlap = abs(np.dot(u, mode_vec))
        contribution = overlap**2 * 100

        freqs.append(freq)
        overlaps.append(overlap)
        contributions.append(contribution)

        logger.log(f"{i+1:<5} | {freq:<12.2f} | {abs(overlap):<10.4f} | {contribution:<5.1f}%")

        if overlap > best_overlap:
            best_overlap = overlap
            winner_freq = freq
            
    logger.log("-" * 48)
    logger.log(f"Hessian Source:        Step {reactant_idx} (Reactant)")
    logger.log(f"Gradient Source:       Step {grad_source_idx}")
    logger.log(f"RMS Force:             {final_rms:.2e}")
    logger.log(f"Attack Freq:           {winner_freq:.2f} cm^-1")
    logger.log("-" * 48)
    

    return {
        'winner_freq': winner_freq, 
        'freqs': freqs, 
        'overlaps': overlaps, 
        'contributions': contributions, 
        'hessian_idx': reactant_idx, 
        'gradient_idx': grad_source_idx, 
        'rms_force': final_rms,
    }








def integrate_action_integral(path, turning_point, raw_energies = False, logger=None):

    
    react_path_m = path.reaction_path 

    if raw_energies:
        wkb_energies = path.energies 
        logger.log("WARNING! Using nonprojected energies, mind you...")
    else:
        wkb_energies = path.proj_energies 
    
    # get V minus E
    V_minus_E = wkb_energies - turning_point 

    # Spline interpolation of V minus E 
    # interpolating the whole curve to avoid spline instability
    # [TODO] maybe we should also plot the splined curve to check for oscillations...
    interpol_VmE = interpolate(react_path_m, V_minus_E) 

    roots = interpol_VmE.roots() 
    roots = np.sort(roots)

    if len(roots) < 2:
        logger.log("Warning: Less than 2 turning points found. That means problems! Check the curves!.")
        action_integral = 0.0
        raise ValueError("Not enough roots found for action integral calculation.")

    if len(roots) > 2:
        logger.log("Warning: More than 2 turning points found. This is unexpected for a simple DWP.",
                   "Proceeding with the first two roots, but this indicates a problem! Check the curves!!!")
        # We will still proceed with the first two roots, but this might indicate a problem. Pre-reactive complex?

    r_start = roots[0]
    r_end = roots[1]
    
    logger.log('Roots = {}'.format(roots))

    # helper funtion
    def wkb_integrand(x):
        val = interpol_VmE(x)

        # only use above zero
        if val < 0:
            return 0.0
        return np.sqrt(val)

    # quad handles the vertical derivatives at r_start and r_end automatically
    action_integral, action_integral_integration_error = quad(wkb_integrand, r_start, r_end)
    
    positive_rp = np.linspace(r_start, r_end, 2000)[1:-2] # positive reaction path
    sqroot = np.sqrt(interpol_VmE(positive_rp)) # squareroot of the interpolated V minus E curve to plot

    return {
    'action_integral': action_integral,
    'action_integral_error': action_integral_integration_error,
    'roots': roots,
    'wkb_energies': wkb_energies,
    'V_minus_E': V_minus_E,
    'positive_rp': positive_rp,
    'sqroot': sqroot
    }








def calc_tunneling_wkb(path,
                   reactant_file_name='',
                   turning_point=None,
                   relevant_vib=None, 
                    software='orca', 
                    raw_energies=False,  
                    dwp=False, 
                    reduced_mass=1.0, 
                    rms_threshold=1e-3,
                    gradient_source=1,
                    kinetic_rate=False,
                    verbose=True):
    """
    Calculates tunneling frequency using WKB approximation.
    
    If 'relevant_vib' is not provided, it attempts to use 'self.relevant_vib'
    calculated via 'calculate_relevant_vib()'.
    """
    logger = logging.GeneralLogger(verbose=verbose) # verbose is defined here and used throughout

    if not turning_point:
        if not reactant_file_name:
            logger.log('WARNING: No prior turning point, so it will be obtained from the path first point.',)
            turning_point = path.energies[0] + projections.calculate_zpe(path.path_frequencies[0])
            logger.log(f"Turning point obtained succesfully! x1 = {turning_point:.6f} Eh")

        # taken out because now first point is always the reactant
        # else:
        #     if verbose: print('WARNING: No turning point found... getting from file...')
        #     self.get_turning_point(software=software, verbose=verbose)

    # moot as the relevant_vib is calculated every time
    # if relevant_vib is None:
        # Check if it was already calculated
        # if self.relevant_vib == 0:
    analysis = calculate_relevant_vib(path,
                                      rms_threshold=rms_threshold, 
                                      gradient_source=gradient_source, 
                                      logger=logger)
        # self.relevant_vib = self.attack_frequency 
    
    vib_analysis = tunnel_results.VibAnalysis(
        label = 'Free_particle',
        winner_freq   = analysis['winner_freq'],
        freqs         = analysis['freqs'],
        overlaps      = analysis['overlaps'],
        contributions = analysis['contributions'],
        hessian_idx   = analysis['hessian_idx'],
        gradient_idx  = analysis['gradient_idx'],
        rms_force     = analysis['rms_force']
)

    logger.log(f"Using calculated Attack Frequency: {vib_analysis.winner_freq:.2f} cm^-1")
    
    # else:
    #     self.relevant_vib = relevant_vib

    integrated_out = integrate_action_integral(
        path=path, 
        turning_point=turning_point, 
        raw_energies=raw_energies, 
        logger=logger)
    
    logger.log(f"Integration limits: {integrated_out['roots'][0]:.2e} to {integrated_out['roots'][1]:.2e} m",
        f"Action Integral: {integrated_out['action_integral']:.4e} (Error est: {integrated_out['action_integral_error']:.2e})")

    tunneling_frequency, theta, permeability, pre_exp = calc_tunneling_frequency(
        relevant_vib=vib_analysis.winner_freq,
        integrated_out=integrated_out,
        reduced_mass=reduced_mass, 
        logger=logger)

    half_life = general.calc_half_life(tunneling_frequency, logger=logger)

    logger.log(f'Tunneling frequency is {tunneling_frequency:.3e} s-1')

    results = tunnel_results.WKBResult(
    vib=vib_analysis,
    action_integral=integrated_out['action_integral'],
    action_integral_error=integrated_out['action_integral_error'],
    theta=theta,
    permeability=permeability,
    pre_exponential=pre_exp,
    roots=integrated_out['roots'],
    turning_point=turning_point,
    V_minus_E=integrated_out['V_minus_E'],
    positive_rp=integrated_out['positive_rp'],     
    sqroot=integrated_out['sqroot'],
    reduced_mass=reduced_mass,      
    raw_energies=raw_energies, 
    wkb_energies=integrated_out['wkb_energies'],      
    tunneling_frequency=tunneling_frequency,
    half_life=half_life,
    exec_logs=logger
    )

    return results








def calc_tunneling_frequency(integrated_out, relevant_vib, reduced_mass=1.0, logger=None):
    """
    Calculates the WKB tunneling frequency using the au-native action integral.
    """
    # WKB Barrier Permeability for DWP: P = 1 / (1 + exp(2*Theta)
    # where Theta = (sqrt(2*m) / hbar) * integral from x1 to x2 of sqrt(2*m*(V(x)-E)) dx
    # Theta = S * sqrt(2*m) / hbar
    
    theta = np.sqrt(2.0 * reduced_mass * const.amu_to_au) * integrated_out['action_integral']
    permeability = 1.0 / (1.0 + np.exp(2.0 * theta))

    logger.log(f'Action integral (au) = {integrated_out['action_integral']:.4e}\n',
               f'WKB Exponent (Theta) = {theta:.4f}\n',
               f'Barrier Permeability = {permeability:.4e}\n')

    # self.relevant_vib is in cm-1. Convert to Hz (s^-1) for the pre-exponential factor
    pre_exp = relevant_vib * const.speed_light
    
    tunneling_frequency = pre_exp * permeability

    return tunneling_frequency, theta, permeability, pre_exp








def calc_tunneling_wkb_sym(path,
                   reactant_file_name='',
                   turning_point=None,
                   relevant_vib=None, 
                    software='orca', 
                    raw_energies=False,  
                    reduced_mass=1.0, 
                    rms_threshold=1e-3,
                    gradient_source=1,
                    kinetic_rate=False,
                    verbose=True):
    """
    Calculates tunneling frequency for a DWP using WKB approximation.
    
    If 'relevant_vib' is not provided, it attempts to use 'self.relevant_vib'
    calculated via 'calculate_relevant_vib()'.
    """
    logger = logging.GeneralLogger(verbose=verbose) # verbose is defined here and used throughout

    if not turning_point:
        if not reactant_file_name:
            logger.log('WARNING: No prior turning point, so it will be obtained from the path first point.')
            turning_point = path.energies[0] + projections.calculate_zpe(path.path_frequencies[0])
            logger.log(f"Turning point obtained succesfully! x1 = {turning_point:.6f} Eh")
        # taken out because now first point is always the reactant
        # else:
        #     if verbose: print('WARNING: No turning point found... getting from file...')
        #     self.get_turning_point(software=software, verbose=verbose)

    # moot as the relevant_vib is calculated every time
    # if relevant_vib is None:
        # Check if it was already calculated
        # if self.relevant_vib == 0:
    analysis = calculate_relevant_vib(path,
                                          rms_threshold=rms_threshold, 
                                          gradient_source=gradient_source, 
                                          logger=logger)
    
    vib_analysis = tunnel_results.VibAnalysis(
        label = 'Free_particle',
        winner_freq   = analysis['winner_freq'],
        freqs         = analysis['freqs'],
        overlaps      = analysis['overlaps'],
        contributions = analysis['contributions'],
        hessian_idx   = analysis['hessian_idx'],
        gradient_idx  = analysis['gradient_idx'],
        rms_force     = analysis['rms_force']
)
    
    # if verbose: print(f"Using calculated Attack Frequency: {relevant_vib:.2f} cm^-1")
    logger.log(f"Using calculated Attack Frequency: {vib_analysis.winner_freq:.2f} cm^-1")
    
    # else:
    #     self.relevant_vib = relevant_vib

    integrated_out = integrate_action_integral(
        path=path, 
        turning_point=turning_point, 
        raw_energies=raw_energies, 
        logger=logger)
    
    logger.log(f"Integration limits: {integrated_out['roots'][0]:.2e} to {integrated_out['roots'][1]:.2e} m",
        f"Action Integral: {integrated_out['action_integral']:.4e} (Error est: {integrated_out['action_integral_error']:.2e})")

    tunneling_frequency, theta, permeability, pre_exp = calc_tunneling_frequency_dwp_wkb_sym(
        relevant_vib=vib_analysis.winner_freq,
        action_integral=integrated_out['action_integral'],
        reduced_mass=reduced_mass, 
        kinetic_rate=kinetic_rate, 
        logger=logger)

    splitting = tunneling_frequency/const.speed_light

    half_life = general.calc_half_life(tunneling_frequency, logger=logger)

    logger.log(f'Tunneling frequency is {tunneling_frequency:.3e} s-1',
        f'Energy level splitting is {tunneling_frequency/const.speed_light:.3e} cm-1')

    results = tunnel_results.WKBSymDWPResult(
    vib=vib_analysis,
    action_integral=integrated_out['action_integral'],
    action_integral_error=integrated_out['action_integral_error'],
    theta=theta,
    permeability=permeability,
    pre_exponential=pre_exp,
    roots=integrated_out['roots'],
    turning_point=turning_point,
    V_minus_E=integrated_out['V_minus_E'],
    positive_rp=integrated_out['positive_rp'],     
    sqroot=integrated_out['sqroot'],
    reduced_mass=reduced_mass,      
    raw_energies=raw_energies,       
    tunneling_frequency=tunneling_frequency,
    half_life=half_life,
    level_splitting=splitting,
    kinetic_rate=kinetic_rate,
    wkb_energies=integrated_out['wkb_energies'],
    exec_logs=logger
    )

    return results








def calc_tunneling_frequency_dwp_wkb_sym(action_integral,
                                         relevant_vib,
                                         reduced_mass, 
                                         kinetic_rate=False,
                                         logger=None):
    # WKB Barrier Permeability for DWP: P = exp(-Theta)
    # where Theta = (sqrt(2*m) / hbar) * integral from x1 to x2 of sqrt(2*m*(V(x)-E)) dx
    # Theta = S * sqrt(2*m) / hbar
    
    reduced_mass = reduced_mass * const.amu_to_au

    # Theta = integral / hbar
    theta = action_integral * np.sqrt(2.0  * reduced_mass) / const.hbar_au

    logger.log(f'Action integral (S) = {action_integral:.4e}')
    logger.log(f'WKB Exponent (Theta) = {theta:.4f}')

    exponential = np.exp(-theta)
    nu_0 = relevant_vib * const.speed_light
    # [FIX] The formula nu = (2 * nu_0 / pi) * exp(-Theta) is just for transposing the barrier
    # the formula for going over the barrier and coming back is (nu_0 / pi) * exp(-Theta)
    # self.tunneling_frequency = (2 * nu_0 / np.pi) * exponential 
    base_freq = (nu_0 / np.pi) * exponential  # Spectroscopic (full period)

    if kinetic_rate:
        tunneling_frequency = 2 * base_freq
    else:
        tunneling_frequency = base_freq

    return tunneling_frequency, theta, exponential, nu_0 / np.pi








def calc_tunneling_wkb_asym(path,
                            reduced_mass=1.0, 
                            rms_threshold=1e-3,
                            gradient_source=1,
                            kinetic_rate=False,
                            temperature=3,
                            verbose=True):
    """
    Calculates tunneling using Rastelli's asymmetric formula (2012).
    Uses path inversion to reuse the attack frequency logic for the product side.
    """
    logger = logging.GeneralLogger(verbose=verbose) # verbose is defined here and used throughout

    reduced_mass = reduced_mass * const.amu_to_au

    # reactant side
    # if self.relevant_vib == 0: # if tested remove later
    logger.log("Calculating Reactant Attack Frequency...")
    vibrational_analysis = calculate_relevant_vib(path, rms_threshold, gradient_source=gradient_source, logger=logger)
    react_analysis = tunnel_results.VibAnalysis(
        label = 'Reactant',
        winner_freq   = vibrational_analysis['winner_freq'],
        freqs         = vibrational_analysis['freqs'],
        overlaps      = vibrational_analysis['overlaps'],
        contributions = vibrational_analysis['contributions'],
        hessian_idx   = vibrational_analysis['hessian_idx'],
        gradient_idx  = vibrational_analysis['gradient_idx'],
        rms_force     = vibrational_analysis['rms_force']
)
    nu_L_cm = vibrational_analysis['winner_freq']
    
    # product side
    logger.log("Calculating Product Attack Frequency (via Path Inversion)...")
    vibrational_analysis = calculate_relevant_vib(path.reversed(), rms_threshold, gradient_source=gradient_source,logger=logger)
    prod_analysis = tunnel_results.VibAnalysis(
            label = 'Reactant',
            winner_freq   = vibrational_analysis['winner_freq'],
            freqs         = vibrational_analysis['freqs'],
            overlaps      = vibrational_analysis['overlaps'],
            contributions = vibrational_analysis['contributions'],
            hessian_idx   = vibrational_analysis['hessian_idx'],
            gradient_idx  = vibrational_analysis['gradient_idx'],
            rms_force     = vibrational_analysis['rms_force']
)
    nu_R_cm = vibrational_analysis['winner_freq']

    # self._reverse_path_data() # flip the reaction_path to use the reactant algorithm in the product
    # try:
    #     nu_R_cm = calculate_relevant_vib(self.path, rms_threshold, gradient_source=gradient_source,verbose=False)
    # finally:
    #     self._reverse_path_data() # FLIP BACK

    if nu_R_cm == 0.0:
        logger.log("Warning: Product frequency calculation failed. Check the curve and the frequencies!!")
        # nu_R_cm = nu_L_cm
        raise ValueError("Product frequency calculation failed.")
    
    if verbose:
        logger.log(f"Reactant attack frequency: {nu_L_cm:.2f} cm^-1",
                   f"Product attack frequency: {nu_R_cm:.2f} cm^-1")

    # frequencies hz
    nu_L_hz = nu_L_cm * const.speed_light
    nu_R_hz = nu_R_cm * const.speed_light

    # zpve hartree
    zpe_L_h = projections.calculate_zpe(path.path_frequencies[0]) #0.5 * nu_L_cm * const.cm_to_hartree
    zpe_R_h = projections.calculate_zpe(path.path_frequencies[-1]) #0.5 * nu_R_cm * const.cm_to_hartree

    # Check if now corrected and remove comments
    E_profile_h = path.proj_energies # * const.hartree_to_J
    E_L = path.energies[0] + zpe_L_h #E_profile_h[0] + zpe_L_h
    E_R = path.energies[-1] + zpe_R_h #E_profile_h[-1] + zpe_R_h

    s_m = path.reaction_path #* const.bohr_to_ang * 1e-10
    
    # TS
    idx_ts = np.argmax(E_profile_h)
    s_ts = s_m[idx_ts]
    V_ts = E_profile_h[idx_ts]

    # check if below the barrier for tunneling
    if E_L >= V_ts or E_R >= V_ts:
        print("Error: Tunneling energy exceeds barrier height. Tunneling is not at place anymore.")
        path.tunneling_frequency = 0.0
        raise ValueError("Energy above barrier.")

    
    V_spline = interpolate(s_m, E_profile_h, k=3)
    theta_prefactor = np.sqrt(2 * reduced_mass) / const.hbar_au

    # reactant -> TS
    try:
        x_tl = brentq(lambda x: V_spline(x) - E_L, s_m[0], s_ts)
    except ValueError:
        x_tl = s_m[0]

    def integrand_L(x): return np.sqrt(max(0, V_spline(x) - E_L))
    S_L, _ = quad(integrand_L, x_tl, s_ts)
    theta_L = (2 * S_L) * theta_prefactor 

    # TS -> product
    try:
        x_tr = brentq(lambda x: V_spline(x) - E_R, s_ts, s_m[-1])
    except ValueError:
        x_tr = s_m[-1]

    def integrand_R(x): return np.sqrt(max(0, V_spline(x) - E_R))
    S_R, _ = quad(integrand_R, s_ts, x_tr)
    theta_R = (2 * S_R) * theta_prefactor 

    
    if kinetic_rate:
        freq_sym_L = (2 * nu_L_hz / np.pi) * np.exp(-theta_L) 
        freq_sym_R = (2 * nu_R_hz / np.pi) * np.exp(-theta_R)
    else:
        freq_sym_L = (nu_L_hz / np.pi) * np.exp(-theta_L) # in s-1
        freq_sym_R = (nu_R_hz / np.pi) * np.exp(-theta_R) # in s-1

    # asymmetry factor A(eta) [Eq 2]
    U_L = V_ts - E_L
    U_R = V_ts - E_R
    
    if U_L <= 0 or U_R <= 0:
        A_eta = 1.0
    else:
        ratio = U_L / U_R
        A_eta = 0.5 * ( (ratio)**0.25 + (1.0/ratio)**0.25 )

    # final frequency [Eq 1]
    # self.tunneling_frequency = A_eta * np.sqrt(freq_sym_L * freq_sym_R)

    # Calculating the splitting
    delta_epsilon = abs(E_R - E_L)
    tun_amplitude = A_eta * np.sqrt(freq_sym_L * freq_sym_R) / 2  # in s-1 (because of the frequency on symm wkb)
    tun_amplitude = tun_amplitude / const.speed_light * const.cm_to_hartree # change back to hartree as is the standard in the code
    level_splitting = np.sqrt( delta_epsilon ** 2 + (2 * tun_amplitude)**2 ) # both in hartree

    # Convert amplitude and frequency to SI for the rate equation
    t_J = tun_amplitude * const.hartree_to_J
    omega_L_rad = 2.0 * np.pi * nu_L_hz
    
    # 2. Base intrinsic rate (Gamma = 2*pi * |t|^2 / (hbar^2 * omega_eff))
    Gamma_s = (2.0 * np.pi / (const.hbar**2 * omega_L_rad)) * (t_J**2)

    tunneling_frequency = Gamma_s  # Now correctly represents the irreversible rate
    half_life = np.log(2) / Gamma_s if Gamma_s > 0 else float('inf')
    
    # calculate coupling matrix element (V) in cm^-1
    coupling_cm = (tun_amplitude * const.hartree_to_cm)

    result = tunnel_results.WkbAsymDwpResult(
        reactant_vib = react_analysis,
        product_vib = prod_analysis, 
        U_L = U_L,
        U_R = U_R,
        A_eta = A_eta,
        coupling_cm = coupling_cm,
        tun_amplitude = tun_amplitude,
        level_splitting = level_splitting,
        tunneling_frequency = tunneling_frequency,
        half_life = half_life
    )

    logger.log("-" * 50)
    logger.log("Rastelli (2012) Asymmetric WKB Analysis")
    logger.log("-" * 50)
    logger.log(f"Freq L (Reactant): {nu_L_cm:.2f} cm-1")
    logger.log(f"Freq R (Product):  {nu_R_cm:.2f} cm-1")
    logger.log(f"Barrier Height L:  {U_L * const.hartree_to_kcalmol:.2f} kcal/mol")
    logger.log(f"Barrier Height R:  {U_R * const.hartree_to_kcalmol:.2f} kcal/mol")
    logger.log(f"Asymmetry Factor:  {A_eta:.4f}")
    logger.log(f"Coupling Matrix Element (V): {coupling_cm:.3e} cm^-1")
    logger.log(f"Level splitting: {tunneling_frequency/const.speed_light:.3e} cm^-1")
    logger.log(f"Tunneling rate: {tunneling_frequency:.3e} s^-1")
    logger.log(f"Half-Life: {half_life:.3e} s")
    logger.log("-" * 50)

    return result










    






# class WKBSolverMixin:

    # def integrate_action_integral(self, raw_energies = False, verbose=True):

        
    #     react_path_m = self.reaction_path 

    #     if raw_energies:
    #         self.wkb_energies = self.energies 
    #         print("WARNING! Using nonprojected energies, mind you...")
    #     else:
    #         self.wkb_energies = self.proj_energies 
        
    #     # get V minus E
    #     self.V_minus_E = self.wkb_energies - self.turning_point 

    #     # Spline interpolation of V minus E 
    #     # interpolating the whole curve to avoid spline instability
    #     # [TODO] maybe we should also plot the splined curve to check for oscillations...
    #     self.interpol_VmE = interpolate(react_path_m, self.V_minus_E) 

    #     self.roots = self.interpol_VmE.roots() 
    #     self.roots = np.sort(self.roots)

    #     if len(self.roots) < 2:
    #         if verbose: print("Warning: Less than 2 turning points found. That means problems! Check the curves!.")
    #         self.action_integral = 0.0
    #         raise ValueError("Not enough roots found for action integral calculation.")

    #     if len(self.roots) > 2:
    #         if verbose: 
    #             print("Warning: More than 2 turning points found. This is unexpected for a simple DWP.")
    #             print("Proceeding with the first two roots, but this indicates a problem! Check the curves!!!")
    #         # We will still proceed with the first two roots, but this might indicate a problem. Pre-reactive complex?

    #     r_start = self.roots[0]
    #     r_end = self.roots[1]
        
    #     if verbose:
    #         print('Roots = {}'.format(self.roots))

    #     # helper funtion
    #     def wkb_integrand(x):
    #         val = self.interpol_VmE(x)

    #         # only use above zero
    #         if val < 0:
    #             return 0.0
    #         return np.sqrt(val)

    #     # quad handles the vertical derivatives at r_start and r_end automatically
    #     self.action_integral, self.action_integral_integration_error = quad(wkb_integrand, r_start, r_end)
        
    #     self.positive_rp = np.linspace(r_start, r_end, 2000)[1:-2] # positive reaction path
    #     self.sqroot = np.sqrt(self.interpol_VmE(self.positive_rp)) # squareroot of the interpolated V minus E curve to plot








    # def calc_tunneling_frequency(self, reduced_mass=1.0, verbose=True):
    #     """
    #     Calculates the WKB tunneling frequency using the au-native action integral.
    #     """
    #     # WKB Barrier Permeability for DWP: P = 1 / (1 + exp(2*Theta)
    #     # where Theta = (sqrt(2*m) / hbar) * integral from x1 to x2 of sqrt(2*m*(V(x)-E)) dx
    #     # Theta = S * sqrt(2*m) / hbar
        
    #     theta = np.sqrt(2.0 * reduced_mass * const.amu_to_au) * self.action_integral
    #     permeability = 1.0 / (1.0 + np.exp(2.0 * theta))
    #     if verbose:
    #         print(f'Action integral (au) = {self.action_integral:.4e}')
    #         print(f'WKB Exponent (Theta) = {theta:.4f}')
    #         print(f'Barrier Permeability = {permeability:.4e}')

    #     # self.relevant_vib is in cm-1. Convert to Hz (s^-1) for the pre-exponential factor
    #     pre_exp = self.relevant_vib * const.speed_light
        
    #     self.tunneling_frequency = pre_exp * permeability

    #     # self.kinetic_rate = True








    # def calc_tunneling_frequency_dwp_wkb_sym(self, 
    #                                          reduced_mass, 
    #                                          kinetic_rate=False,
    #                                          verbose=True):
    #     # WKB Barrier Permeability for DWP: P = exp(-Theta)
    #     # where Theta = (sqrt(2*m) / hbar) * integral from x1 to x2 of sqrt(2*m*(V(x)-E)) dx
    #     # Theta = S * sqrt(2*m) / hbar
        
    #     reduced_mass = reduced_mass * const.amu_to_au

    #     # Theta = integral / hbar
    #     theta = self.action_integral * np.sqrt(2.0  * reduced_mass) / const.hbar_au

    #     if verbose:
    #         print(f'Action integral (S) = {self.action_integral:.4e}')
    #         print(f'WKB Exponent (Theta) = {theta:.4f}')

    #     exponential = np.exp(-theta)
    #     nu_0 = self.relevant_vib * const.speed_light
    #     # [FIX] The formula nu = (2 * nu_0 / pi) * exp(-Theta) is just for transposing the barrier
    #     # the formula for going over the barrier and coming back is (nu_0 / pi) * exp(-Theta)
    #     # self.tunneling_frequency = (2 * nu_0 / np.pi) * exponential 
    #     base_freq = (nu_0 / np.pi) * exponential  # Spectroscopic (full period)

    #     if kinetic_rate:
    #         self.tunneling_frequency = 2 * base_freq
    #     else:
    #         self.tunneling_frequency = base_freq








    # # [TODO] Fix the methods names inside here. Should fix this in general...
    # def calc_tunneling(self, 
    #                    relevant_vib=None, 
    #                    software='gaussian', 
    #                    raw_energies=False,  
    #                    dwp=False, 
    #                    reduced_mass=1, 
    #                    rms_threshold=1e-3,
    #                    gradient_source=1,
    #                 #    symmetric=True,
    #                 #    wkb=False,
    #                 #    aps=True,
    #                    kinetic_rate=False,
    #                 #    find_well=False,
    #                    verbose=True):
    #     """
    #     Calculates tunneling frequency for a DWP using WKB approximation.
        
    #     If 'relevant_vib' is not provided, it attempts to use 'self.relevant_vib'
    #     calculated via 'calculate_relevant_vib()'.
    #     """

    #     if not self.turning_point:
    #         if not self.reactant_file_name:
    #             if verbose: print('WARNING: No prior turning point. Obtaining it from the NEB first point.')
    #             self.turning_point = self.energies[0] + projections.calculate_zpe(self.path_frequencies[0])
    #             if verbose: print(f"Turning point obtained succesfully! x1 = {self.turning_point:.6f} Eh")
    #         else:
    #             if verbose: print('WARNING: No turning point found... getting from file...')
    #             self.get_turning_point(software=software, verbose=verbose)


    #     if relevant_vib is None:
    #         # Check if it was already calculated
    #         if self.relevant_vib == 0:
    #                 self.relevant_vib = calculate_relevant_vib(self.path,
    #                                                            rms_threshold=rms_threshold, 
    #                                                                 gradient_source=gradient_source, 
    #                                                                 verbose=verbose)
    #         # self.relevant_vib = self.attack_frequency 
    #         if verbose: print(f"Using calculated Attack Frequency: {self.relevant_vib:.2f} cm^-1")
    #     else:
    #         self.relevant_vib = relevant_vib


    #     self.integrate_action_integral(raw_energies=raw_energies, verbose=verbose)
        
    #     if verbose:
    #         print(f"Integration limits: {self.roots[0]:.2e} to {self.roots[1]:.2e} m")
    #         print(f"Action Integral: {self.action_integral:.4e} (Error est: {self.action_integral_integration_error:.2e})")
    #         # print(f'Barrier penetration (θ) = {np.sqrt(2 * reduced_mass * const.amu_to_au) * self.action_integral/const.hbar_au:.2f}')
        

    #     if dwp:
    #         # if symmetric:
    #             # if wkb:
    #         self.calc_tunneling_frequency_dwp_wkb_sym(reduced_mass=reduced_mass, kinetic_rate=kinetic_rate, verbose=verbose)
    #             # elif aps:
    #             #     if not find_well:
    #             #         self.calc_tunneling_frequency_aps(verbose=verbose)
    #             #     else:
    #             #         self.calc_tunneling_frequency_aps_find_well(verbose=verbose)   
                    
    #                         # self.tunneling_frequency_from_splitting()
    #     else:
    #         self.calc_tunneling_frequency(reduced_mass=reduced_mass, verbose=verbose)

    #     if verbose:
    #         print('Tunneling frequency is {:.3e} s-1'.format(self.tunneling_frequency))
    #         print('Energy level splitting is {:.3e} cm-1'.format(self.tunneling_frequency/const.speed_light))








